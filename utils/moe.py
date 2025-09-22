import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class MoELayer(nn.Module):
    def __init__(self, input_dim, expert_dim, num_experts, k=1):
        """
        Enhanced MoE Layer with Complex Gating Network
        :param input_dim: 输入维度
        :param expert_dim: 专家输出维度（一般等于input_dim）
        :param num_experts: 初始专家数量
        :param k: top-k routing
        """
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.k = k
        self.input_dim = input_dim
        
        # 专家网络保持不变
        self.experts = nn.ModuleList([
            nn.Linear(input_dim, expert_dim) for _ in range(num_experts)
        ])
        
        # ===== 增强门控网络 =====
        self.gate = self.build_enhanced_gate(input_dim, num_experts)
        
        # ===== 路由历史记忆 =====
        self.routing_memory = nn.Parameter(
            torch.zeros(num_experts, input_dim),
            requires_grad=False  # 不通过梯度更新
        )
        
        # ===== 特征注意力机制 =====
        self.feature_attention = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )
        
        # 保存旧门控权重，用于扩展时初始化
        self._old_gate_weights = None
        
        # 诊断工具：路由准确性历史记录
        self.routing_acc_history = []
        self.routing_confusion_matrices = []
        
        # 诊断工具：专家使用历史记录
        self.expert_usage_history = []
        
        # 诊断工具：门控输出统计
        self.gate_stats = {
            "mean": [],
            "std": [],
            "min": [],
            "max": []
        }

    def build_enhanced_gate(self, input_dim, num_experts):
        """构建更复杂的门控网络"""
        return nn.Sequential(
            nn.Linear(input_dim * 2, 512),  # 输入维度加倍（原始特征+路由记忆）
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_experts)
        )

    def forward(self, x, task_id=None, routing_targets=None):
        """
        x: (B, D)
        task_id: int或torch.Tensor, 可选。若提供，则强制路由到指定专家（用于增量学习控制）
        routing_targets: (B,), 可选。每个样本应该路由到的目标专家ID（用于监督学习）
        """
        B, D = x.shape
        
        # ===== 特征注意力加权 =====
        attn_weights = self.feature_attention(x)
        x_attn = x * attn_weights
        
        # ===== 路由记忆融合 =====
        if routing_targets is not None and self.training:
            # 更新路由记忆（指数移动平均）
            with torch.no_grad():
                for expert_id in range(self.num_experts):
                    expert_mask = (routing_targets == expert_id)
                    if expert_mask.any():
                        expert_features = x_attn[expert_mask].mean(dim=0)
                        self.routing_memory[expert_id] = (
                            0.9 * self.routing_memory[expert_id] +
                            0.1 * expert_features
                        )
        
        # 创建路由记忆输入
        memory_input = torch.index_select(
            self.routing_memory, 
            0, 
            routing_targets if routing_targets is not None else torch.zeros(B, dtype=torch.long, device=x.device)
        )
        
        # 组合输入：原始特征 + 路由记忆
        gate_input = torch.cat([x_attn, memory_input], dim=1)
        
        # ===== 通过增强门控网络 =====
        gate_logits = self.gate(gate_input)
        
        # ===== 计算路由损失 =====
        routing_loss = 0
        if routing_targets is not None and self.training:
            routing_loss = F.cross_entropy(gate_logits, routing_targets)
            
            # 添加专家专业化损失
            spec_loss = self.expert_specialization_loss()
            routing_loss += 0.05 * spec_loss

        # ===== 诊断工具：记录门控输出统计 =====
        if self.training:
            self.gate_stats["mean"].append(gate_logits.mean().item())
            self.gate_stats["std"].append(gate_logits.std().item())
            self.gate_stats["min"].append(gate_logits.min().item())
            self.gate_stats["max"].append(gate_logits.max().item())
        
        # ===== 支持每个样本的任务ID =====
        if task_id is not None:
            # 如果task_id是整数（标量），转换为张量
            if isinstance(task_id, int):
                task_id = torch.full((B,), task_id, dtype=torch.long, device=x.device)
            
            # 创建掩码，只允许每个样本指定的专家被选中
            mask = torch.full_like(gate_logits, float('-inf'))
            mask[torch.arange(B), task_id] = 0
            gate_logits = gate_logits + mask
            
        topk_vals, topk_idxs = torch.topk(gate_logits, self.k, dim=1)  # (B, k)
        topk_vals = F.softmax(topk_vals, dim=1)  # (B, k)
        
        # ===== 诊断工具：计算路由准确性 =====
        if routing_targets is not None and self.training:
            # 获取预测的路由目标（Top1）
            predicted_task_id = topk_idxs[:, 0]  # 取第一个TopK选择
            
            # 计算路由准确性
            routing_accuracy = (predicted_task_id == routing_targets).float().mean()
            self.routing_acc_history.append(routing_accuracy.item())
            
            # 计算混淆矩阵
            if self.num_experts <= 10:  # 避免输出太大
                cm = confusion_matrix(
                    routing_targets.cpu().numpy(), 
                    predicted_task_id.cpu().numpy(),
                    labels=range(self.num_experts)
                )
                self.routing_confusion_matrices.append(cm)

        out = torch.zeros(B, self.experts[0].out_features, device=x.device)

        for i in range(self.num_experts):
            expert_mask = (topk_idxs == i).any(dim=1)  # (B,)
            if expert_mask.any():
                batch_x = x[expert_mask]  # (n, D)
                expert_out = self.experts[i](batch_x)  # (n, D_out)

                # 获取这些样本在topk中分配给专家i的权重
                weights = topk_vals[expert_mask]  # (n, k)
                idx_match = (topk_idxs[expert_mask] == i).float()  # (n, k)
                weighted = (weights * idx_match).sum(dim=1, keepdim=True)  # (n, 1)

                out[expert_mask] += weighted * expert_out
        
        # === 专家使用统计 ===
        expert_counts = torch.zeros(self.num_experts, device=x.device)
        for i in range(self.num_experts):
            expert_mask = (topk_idxs == i).any(dim=1)
            expert_counts[i] = expert_mask.float().sum()
        
        # ===== 诊断工具：记录专家使用情况 =====
        if self.training:
            self.expert_usage_history.append(expert_counts.detach().cpu().numpy())
        
        # 保存专家使用统计供外部访问
        self.last_expert_counts = expert_counts.detach().clone()
        
        return {
            "output": out,
            "routing_loss": routing_loss,  # 返回路由损失
            "gate_logits": gate_logits,
            "expert_assignments": topk_idxs,
            "attention_weights": attn_weights  # 新增：返回注意力权重
        }

    def expert_specialization_loss(self):
        """计算专家专业化损失，鼓励专家差异化"""
        similarities = 0
        count = 0
        
        # 获取专家权重
        expert_weights = [expert.weight for expert in self.experts]
        
        for i in range(len(expert_weights)):
            for j in range(i+1, len(expert_weights)):
                # 计算余弦相似度
                sim = F.cosine_similarity(
                    expert_weights[i].flatten().unsqueeze(0),
                    expert_weights[j].flatten().unsqueeze(0)
                )
                similarities += sim.item()
                count += 1
        
        if count == 0:
            return 0
        
        # 我们希望最小化专家之间的相似度
        avg_similarity = similarities / count
        return avg_similarity

    def expand_experts(self, new_num_experts):
        """动态增加专家数量"""
        if new_num_experts <= self.num_experts:
            return

        old_num = self.num_experts
        input_dim = self.experts[0].in_features
        output_dim = self.experts[0].out_features
        
        # 获取当前设备
        device = next(self.experts[0].parameters()).device

        # === 1. 计算现有专家的平均参数 ===
        with torch.no_grad():
            weights = torch.stack([e.weight.data.clone() for e in self.experts])
            biases = torch.stack([e.bias.data.clone() for e in self.experts])
            
            avg_weight = torch.mean(weights, dim=0).to(device)
            avg_bias = torch.mean(biases, dim=0).to(device)

        # === 2. 添加新专家（使用平均参数初始化）===
        for i in range(old_num, new_num_experts):
            new_expert = nn.Linear(input_dim, output_dim).to(device)
            new_expert.weight.data.copy_(avg_weight)
            new_expert.bias.data.copy_(avg_bias)
            
            # 添加噪声
            noise_weight = torch.randn_like(avg_weight, device=device) * 0.02
            noise_bias = torch.randn_like(avg_bias, device=device) * 0.02
            new_expert.weight.data.add_(noise_weight)
            new_expert.bias.data.add_(noise_bias)
            
            self.experts.append(new_expert)

        # === 3. 扩展路由记忆 ===
        new_memory = torch.zeros(new_num_experts - old_num, self.input_dim, device=device)
        self.routing_memory = nn.Parameter(
            torch.cat([self.routing_memory.data, new_memory], dim=0),
            requires_grad=False
        )

        # === 4. 扩展门控层 ===
        # 保存旧门控权重（只保存Linear层的权重）
        if self._old_gate_weights is None:
            self._old_gate_weights = []
            for module in self.gate:
                if isinstance(module, nn.Linear):
                    self._old_gate_weights.append({
                        'weight': module.weight.data.clone(),
                        'bias': module.bias.data.clone() if module.bias is not None else None
                    })
        
        # 创建新的门控网络
        new_gate = self.build_enhanced_gate(self.input_dim, new_num_experts).to(device)
        
        # 复制旧权重到新门控网络
        with torch.no_grad():
            if self._old_gate_weights:
                # 获取新门控网络中的所有Linear层
                new_linear_layers = [m for m in new_gate if isinstance(m, nn.Linear)]
                
                # 复制权重到前几个Linear层（排除最后一层）
                for i in range(len(self._old_gate_weights) - 1):
                    if i < len(new_linear_layers):
                        new_linear_layers[i].weight.data.copy_(self._old_gate_weights[i]['weight'])
                        if self._old_gate_weights[i]['bias'] is not None:
                            new_linear_layers[i].bias.data.copy_(self._old_gate_weights[i]['bias'])
                
                # 处理最后一层（输出层）
                last_layer = new_linear_layers[-1]
                old_last_layer = self._old_gate_weights[-1]
                
                # 复制旧专家对应的权重
                last_layer.weight.data[:old_num] = old_last_layer['weight']
                if old_last_layer['bias'] is not None:
                    last_layer.bias.data[:old_num] = old_last_layer['bias']
                
                # 初始化新专家门控权重为负值
                last_layer.weight.data[old_num:] = torch.randn_like(last_layer.weight.data[old_num:]) * 0.01 - 1.0
                last_layer.bias.data[old_num:].fill_(-1.0)
        
        self.gate = new_gate
        self.num_experts = new_num_experts

        print(f"✅ MoE expanded to {new_num_experts} experts.")
        