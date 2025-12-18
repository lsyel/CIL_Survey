import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy

class MoELayer(nn.Module):
    """
    在您原有MoELayer基础上添加门控网络蒸馏功能
    通过将蒸馏损失整合到路由损失中，保持上层接口不变
    """
    def __init__(self, input_dim, expert_dim, num_experts, k=1,
                 distill_weight=1, temperature=2.0):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.k = k
        self.distill_weight = distill_weight  # 蒸馏损失权重
        self.temperature = temperature  # 蒸馏温度
        
        # 专家网络（保持不变）
        self.experts = nn.ModuleList([
            nn.Linear(input_dim, expert_dim) for _ in range(num_experts)
        ])
        
        # 门控网络（保持您原有的结构）
        self.gate = self._build_gate_network(input_dim, num_experts)
        
        # 旧门控网络（用于蒸馏）
        self.old_gate = self.gate
        self.old_num_experts = 0
        # 保存旧门控权重用于专家扩展
        self._old_gate_weights = None

    def _build_gate_network(self, input_dim, num_experts):
        """构建门控网络（保持您原有的结构）"""
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_experts)
        )

    def set_old_gate(self, old_gate, old_num_experts):
        """设置旧门控网络用于蒸馏"""
        if old_gate is not None:
            self.old_gate = copy.deepcopy(old_gate)
            self.old_gate.eval()  # 设置为评估模式
            for param in self.old_gate.parameters():
                param.requires_grad = False
            self.old_num_experts = old_num_experts
            print(f"🔁 Loaded old gate with {old_num_experts} experts for distillation")

    def forward(self, x, task_id=None, routing_targets=None):
        """
        前向传播，将蒸馏损失整合到路由损失中
        上层代码完全不需要修改
        """
        B, D = x.shape
        gate_logits = self.gate(x)
        
        # 计算路由损失
        routing_loss = 0
        distill_loss = 0
        total_routing_loss = 0
        
        if routing_targets is not None and self.training:
            # 基础路由损失
            routing_loss = F.cross_entropy(gate_logits, routing_targets)
            
            # 计算蒸馏损失（如果存在旧门控网络）
            if self.old_gate is not None and self.old_num_experts > 0:
                distill_loss = self._compute_distill_loss(x, gate_logits)
                
                # 将蒸馏损失整合到总路由损失中
                total_routing_loss = routing_loss + self.distill_weight * distill_loss
                if random.random() < 0.1:
                    print(f"total_routing_loss: {total_routing_loss.item():.4f}, routing_loss: {routing_loss.item():.4f}, distill_loss: {distill_loss.item():.4f}")
            else:
                total_routing_loss = routing_loss
                # print(f"Routing loss: {routing_loss.item():.4f}")
            
        
        # Top-k 选择
        topk_vals, topk_idxs = torch.topk(gate_logits, self.k, dim=1)
        topk_vals = F.softmax(topk_vals, dim=1)

        # 计算输出（保持不变）
        out = torch.zeros(B, self.experts[0].out_features, device=x.device)
        for i in range(self.num_experts):
            expert_mask = (topk_idxs == i).any(dim=1)
            if expert_mask.any():
                batch_x = x[expert_mask]
                expert_out = self.experts[i](batch_x)
                
                weights = topk_vals[expert_mask]
                idx_match = (topk_idxs[expert_mask] == i).float()
                weighted = (weights * idx_match).sum(dim=1, keepdim=True)
                
                out[expert_mask] += weighted * expert_out
        
        # 返回结果：将总路由损失（包含蒸馏）作为routing_loss返回
        # 上层代码完全不需要修改
        return {
            "output": out,
            "routing_loss": total_routing_loss,  # 关键修改：这里包含了蒸馏损失
            "gate_logits": gate_logits,
            "expert_assignments": topk_idxs
        }

    def _compute_distill_loss(self, x, current_gate_logits):
        """计算门控网络蒸馏损失"""
        with torch.no_grad():
            # 旧门控网络的输出
            old_gate_logits = self.old_gate(x)
            
            # 如果专家数量不同，需要对齐
            if self.old_num_experts < self.num_experts:
                # 填充旧门控输出到当前维度
                expanded_old_logits = torch.zeros_like(current_gate_logits)
                expanded_old_logits[:, :self.old_num_experts] = old_gate_logits
                
                # 对新专家部分使用中性 Logits (0)，这对应于 Softmax 后的均匀概率。
                # 这里保持为0即可，因为 expanded_old_logits 已经初始化为0。
                # expanded_old_logits[:, self.old_num_experts:] = 0.0 # 理论上可以省略，但写出来更清晰
                
                old_gate_logits = expanded_old_logits
            elif self.old_num_experts > self.num_experts:
                # 截断旧门控输出（保持不变）
                old_gate_logits = old_gate_logits[:, :self.num_experts]
        
        # -----------------------------------------------------------
        # 使用KL散度计算蒸馏损失（保持不变，注意温度 T 的应用）
        # LogSoftmax/Softmax 应该作用在完整的 Logits 向量上
        # -----------------------------------------------------------
        current_probs = F.log_softmax(current_gate_logits / self.temperature, dim=1)
        old_probs = F.softmax(old_gate_logits / self.temperature, dim=1)
        
        # 注意：F.kl_div 的 reduction='batchmean' 默认是对所有元素求和后除以 batch size。
        distill_loss = F.kl_div(current_probs, old_probs, reduction='batchmean') * (self.temperature ** 2)
        return distill_loss

    def expand_experts(self, new_num_experts):
        """扩展专家（保存旧门控网络用于蒸馏）"""
        if new_num_experts <= self.num_experts:
            return

        old_num = self.num_experts
        self._freeze_experts(old_num)
        # 保存当前门控网络作为旧门控网络（用于蒸馏）
        self.set_old_gate(self.gate, old_num)
        
        # 原有的专家扩展逻辑
        input_dim = self.experts[0].in_features
        output_dim = self.experts[0].out_features
        device = next(self.experts[0].parameters()).device

        # 添加新专家
        self._add_new_experts(old_num, new_num_experts, input_dim, output_dim, device)
        
        # 扩展门控网络
        self.gate = self._expand_gate_network(new_num_experts, device)
        self.num_experts = new_num_experts
        
        print(f"✅ MoE expanded to {new_num_experts} experts with gate distillation.")
        self.distill_weight+=0.0


    def _add_new_experts(self, old_num, new_num_experts, input_dim, output_dim, device):
        """添加新专家（保持不变）"""
        # 计算所有旧专家的平均值
        with torch.no_grad():
            weights = torch.stack([e.weight.data.clone() for e in self.experts])
            biases = torch.stack([e.bias.data.clone() for e in self.experts])
            
            avg_weight = torch.mean(weights, dim=0).to(device)
            avg_bias = torch.mean(biases, dim=0).to(device)

        # 添加新专家
        for i in range(old_num, new_num_experts):
            new_expert = nn.Linear(input_dim, output_dim).to(device)
            new_expert.weight.data.copy_(avg_weight)
            new_expert.bias.data.copy_(avg_bias)
            
            # 添加噪声
            noise_scale = 0.01
            new_expert.weight.data.add_(torch.randn_like(avg_weight) * noise_scale)
            new_expert.bias.data.add_(torch.randn_like(avg_bias) * noise_scale)
            
            self.experts.append(new_expert)

    def _expand_gate_network(self, new_num_experts, device):
        """扩展门控网络（保持不变）"""
        # 保存旧门控权重（如果尚未保存）
        if self._old_gate_weights is None:
            self._old_gate_weights = []
            for layer in self.gate:
                if isinstance(layer, nn.Linear):
                    self._old_gate_weights.append({
                        'weight': layer.weight.data.clone(),
                        'bias': layer.bias.data.clone()
                    })

        # 构建新门控网络
        new_gate = self._build_gate_network(self.experts[0].in_features, new_num_experts).to(device)
        
        # 复制旧权重到新门控网络
        self._copy_gate_weights(new_gate, new_num_experts)
        
        return new_gate

    def _copy_gate_weights(self, new_gate, new_num_experts):
        """复制旧门控网络的权重到新门控网络（保持不变）"""
        if self._old_gate_weights is None:
            return
            
        with torch.no_grad():
            # 复制前4层（与旧网络相同）
            new_gate[0].weight.data.copy_(self._old_gate_weights[0]['weight'])
            new_gate[0].bias.data.copy_(self._old_gate_weights[0]['bias'])
            
            new_gate[3].weight.data.copy_(self._old_gate_weights[1]['weight'])
            new_gate[3].bias.data.copy_(self._old_gate_weights[1]['bias'])
            
            # 复制新增的层（第5层）
            if len(self._old_gate_weights) > 4:
                new_gate[6].weight.data.copy_(self._old_gate_weights[4]['weight'])
                new_gate[6].bias.data.copy_(self._old_gate_weights[4]['bias'])
            
            # 处理输出层
            old_output_dim = self._old_gate_weights[-1]['weight'].size(0)
            new_gate[-1].weight.data[:old_output_dim] = self._old_gate_weights[-1]['weight']
            new_gate[-1].bias.data[:old_output_dim] = self._old_gate_weights[-1]['bias']
            
            if new_num_experts > old_output_dim:
                # 新专家部分：使用平均值初始化
                avg_weight = torch.mean(self._old_gate_weights[-1]['weight'], dim=0)
                avg_bias = torch.mean(self._old_gate_weights[-1]['bias'], dim=0)
                
                for i in range(old_output_dim, new_num_experts):
                    new_gate[-1].weight.data[i] = avg_weight.clone()
                    new_gate[-1].bias.data[i] = avg_bias.clone()
                    
                    # 添加小量噪声
                    noise_scale = 0.01
                    new_gate[-1].weight.data[i].add_(
                        torch.randn_like(avg_weight) * noise_scale
                    )
                    new_gate[-1].bias.data[i].add_(
                        torch.randn_like(avg_bias) * noise_scale
                    )
    def _freeze_experts(self, num_to_freeze):
            """
            冻结前 k 个专家的参数
            """
            for i in range(num_to_freeze):
                expert = self.experts[i]
                # 设为评估模式 (影响 Dropout/BatchNorm)
                expert.eval() 
                # 关闭梯度计算
                for param in expert.parameters():
                    param.requires_grad = False
            
            print(f"🔒 Frozen {num_to_freeze} experts.")