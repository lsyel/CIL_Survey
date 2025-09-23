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
        MoE Layer with Softmax Gating and Comprehensive Diagnostics
        :param input_dim: 输入维度
        :param expert_dim: 专家输出维度（一般等于input_dim）
        :param num_experts: 初始专家数量
        :param k: top-k routing
        """
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.k = k
        self.experts = nn.ModuleList([
            nn.Linear(input_dim, expert_dim) for _ in range(num_experts)
        ])
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_experts)
        )
        # 保存旧门控权重，用于扩展时初始化
        self._old_gate_weights = None
        self._old_gate_bias = None
        
        # 诊断工具：路由准确性历史记录
        self.routing_acc_history = []
        self.routing_confusion_matrices = []
        
        # 诊断工具：专家使用历史记录
        self.expert_usage_history = []

    def forward(self, x, task_id=None, routing_targets=None):
        """
        x: (B, D)
        task_id: int或torch.Tensor, 可选。若提供，则强制路由到指定专家（用于增量学习控制）
        routing_targets: (B,), 可选。每个样本应该路由到的目标专家ID（用于监督学习）
        """
        B, D = x.shape
        gate_logits = self.gate(x)  # (B, E)
        
        # ===== 关键修改：计算路由损失 =====
        routing_loss = 0
        if routing_targets is not None and self.training:
            # 计算路由损失（交叉熵损失）
            routing_loss = F.cross_entropy(gate_logits, routing_targets)

        
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
        if routing_targets  is not None and self.training:
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
            "expert_assignments": topk_idxs
        }

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
            noise_weight = torch.randn_like(avg_weight, device=device) * 0.01
            noise_bias = torch.randn_like(avg_bias, device=device) * 0.01
            new_expert.weight.data.add_(noise_weight)
            new_expert.bias.data.add_(noise_bias)
            
            self.experts.append(new_expert)

        # === 3. 保存当前门控权重 ===
        if self._old_gate_weights is None:
            # 保存Sequential中所有Linear层的权重和偏置
            self._old_gate_weights = []
            for layer in self.gate:
                if isinstance(layer, nn.Linear):
                    self._old_gate_weights.append({
                        'weight': layer.weight.data.clone(),
                        'bias': layer.bias.data.clone()
                    })

        # === 4. 扩展门控层 ===
        # 创建新的门控网络（Sequential）
        new_gate = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, new_num_experts)
        ).to(device)
        
        with torch.no_grad():
            # 复制旧门控权重（Sequential中的前两个Linear层）
            if self._old_gate_weights is not None:
                # 第一个Linear层（输入层到隐藏层）
                new_gate[0].weight.data.copy_(self._old_gate_weights[0]['weight'])
                new_gate[0].bias.data.copy_(self._old_gate_weights[0]['bias'])
                
                # 第二个Linear层（隐藏层到输出层）
                # 注意：输出维度扩展了
                old_output_dim = self._old_gate_weights[1]['weight'].size(0)
                new_gate[2].weight.data[:old_output_dim] = self._old_gate_weights[1]['weight']
                new_gate[2].bias.data[:old_output_dim] = self._old_gate_weights[1]['bias']
                
                # 新专家门控初始化为负偏置
                if new_num_experts > old_num:
                    # 初始化新门控权重
                    new_gate[2].weight[old_output_dim:].normal_(mean=-0.1, std=0.01)
                    new_gate[2].bias[old_output_dim:].fill_(-0.5)  # 负偏置
        
        self.gate = new_gate
        self.num_experts = new_num_experts

        print(f"✅ MoE expanded to {new_num_experts} experts.")
        
        # 重置诊断数据
        self.reset_diagnostics()
    
    # ===== 诊断工具：路由准确性报告 =====
    def report_routing_accuracy(self, window_size=100):
        """报告路由准确性统计"""
        if not self.routing_acc_history:
            return None
        
        # 计算平均准确性
        avg_acc = np.mean(self.routing_acc_history)
        
        # 计算最近window_size个样本的准确性
        recent_acc = np.mean(self.routing_acc_history[-window_size:]) if len(self.routing_acc_history) >= window_size else avg_acc
        
        # 计算准确性趋势（最近一半 vs 前一半）
        half = len(self.routing_acc_history) // 2
        if half > 0:
            first_half = np.mean(self.routing_acc_history[:half])
            second_half = np.mean(self.routing_acc_history[half:])
            trend = "improving" if second_half > first_half else "declining" if second_half < first_half else "stable"
        else:
            trend = "unknown"
        
        # 计算混淆矩阵（平均）
        if self.routing_confusion_matrices:
            avg_cm = np.mean(self.routing_confusion_matrices, axis=0)
        else:
            avg_cm = None
        
        return {
            "average_accuracy": avg_acc,
            "recent_accuracy": recent_acc,
            "accuracy_trend": trend,
            "confusion_matrix": avg_cm
        }
    
    # ===== 诊断工具：专家使用报告 =====
    def report_expert_usage(self):
        """报告专家使用统计"""
        if not self.expert_usage_history:
            return None
        
        # 计算平均使用情况
        avg_usage = np.mean(self.expert_usage_history, axis=0)
        
        # 计算不平衡度
        min_usage = avg_usage.min()
        max_usage = avg_usage.max()
        imbalance_ratio = max_usage / (min_usage + 1e-6)
        
        # 计算使用趋势（最近一半 vs 前一半）
        half = len(self.expert_usage_history) // 2
        if half > 0:
            first_half = np.mean(self.expert_usage_history[:half], axis=0)
            second_half = np.mean(self.expert_usage_history[half:], axis=0)
            trends = ["improving" if sh > fh else "declining" if sh < fh else "stable" for fh, sh in zip(first_half, second_half)]
        else:
            trends = ["unknown"] * self.num_experts
        
        return {
            "average_usage": avg_usage,
            "imbalance_ratio": imbalance_ratio,
            "usage_trends": trends
        }
    # ===== 诊断工具：重置诊断数据 =====
    def reset_diagnostics(self):
        """重置所有诊断数据"""
        self.routing_acc_history = []
        self.routing_confusion_matrices = []
        self.expert_usage_history = []
        self.gate_stats = {
            "mean": [],
            "std": [],
            "min": [],
            "max": []
        }
    