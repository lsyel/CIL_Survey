import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class MoELayer(nn.Module):
    def __init__(self, input_dim, expert_dim, num_experts, k=1, print_prob=0.003):
        """
        MoE Layer with Conditional Statistics Recording
        :param print_prob: 打印概率，为0时不记录任何统计信息
        """
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.k = k
        self.print_prob = print_prob
        
        self.experts = nn.ModuleList([
            nn.Linear(input_dim, expert_dim) for _ in range(num_experts)
        ])
        
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_experts)
        )
        
        # 保存旧门控权重
        self._old_gate_weights = None
        self._old_gate_bias = None
        
        # 仅当print_prob>0时初始化统计变量
        if print_prob > 0:
            self.routing_acc_history = []
            self.expert_usage_history = []
        else:
            # 设置为None，避免不必要的内存分配
            self.routing_acc_history = None
            self.expert_usage_history = None

    def forward(self, x, task_id=None, routing_targets=None):
        B, D = x.shape
        gate_logits = self.gate(x)
        
        # 计算路由损失
        routing_loss = 0
        if routing_targets is not None and self.training:
            routing_loss = F.cross_entropy(gate_logits, routing_targets)
            
        topk_vals, topk_idxs = torch.topk(gate_logits, self.k, dim=1)
        topk_vals = F.softmax(topk_vals, dim=1)
        
        # 仅当print_prob>0且需要路由目标时才计算路由准确性
        routing_accuracy = 0
        should_record = (self.print_prob > 0) and (routing_targets is not None) and self.training
        
        if should_record:
            predicted_task_id = topk_idxs[:, 0]
            routing_accuracy = (predicted_task_id == routing_targets).float().mean()
            self.routing_acc_history.append(routing_accuracy.item())
            
            # 随机打印路由信息
            if random.random() < self.print_prob:
                self._print_simple_info(routing_accuracy, routing_loss, topk_idxs, routing_targets)

        # 计算输出
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
        
        # 仅当需要记录时才计算专家使用统计
        expert_counts = None
        if should_record:
            expert_counts = torch.zeros(self.num_experts, device=x.device)
            for i in range(self.num_experts):
                expert_mask = (topk_idxs == i).any(dim=1)
                expert_counts[i] = expert_mask.float().sum()
            
            self.expert_usage_history.append(expert_counts.detach().cpu().numpy())
            self.last_expert_counts = expert_counts.detach().clone()
        
        return {
            "output": out,
            "routing_loss": routing_loss,
            "gate_logits": gate_logits,
            "expert_assignments": topk_idxs
        }

    def _print_simple_info(self, routing_accuracy, routing_loss, topk_idxs, routing_targets):
        """简单打印路由信息"""
        print(f"\n🎯 MoE Info (Random Print, prob={self.print_prob}):")
        print(f"  Routing Accuracy: {routing_accuracy.item():.4f}")
        print(f"  Routing Loss: {routing_loss.item() if isinstance(routing_loss, torch.Tensor) else routing_loss:.4f}")
        
        # 专家使用情况
        if hasattr(self, 'last_expert_counts') and self.last_expert_counts is not None:
            expert_counts = self.last_expert_counts.cpu().numpy()
            print("  Expert Usage:")
            for i, count in enumerate(expert_counts):
                print(f"    Expert {i}: {count:.0f} samples")
        
        # 路由分布
        if routing_targets is not None:
            target_dist = torch.bincount(routing_targets, minlength=self.num_experts).cpu().numpy()
            pred_dist = torch.bincount(topk_idxs[:, 0], minlength=self.num_experts).cpu().numpy()
            
            print("  Routing Distribution:")
            # 只显示有样本的专家
            displayed_experts = 0
            for i in range(self.num_experts):
                if target_dist[i] > 0 or pred_dist[i] > 0:
                    print(f"    Expert {i}: Target={target_dist[i]}, Predicted={pred_dist[i]}")
        
        # 历史准确性（最近10个batch的平均）
        if self.routing_acc_history and len(self.routing_acc_history) >= 10:
            recent_acc = np.mean(self.routing_acc_history[-10:])
            print(f"  Recent Avg Accuracy: {recent_acc:.4f}")
        
        print("-" * 40)

    def expand_experts(self, new_num_experts):
        """扩展专家"""
        if new_num_experts <= self.num_experts:
            return

        old_num = self.num_experts
        input_dim = self.experts[0].in_features
        output_dim = self.experts[0].out_features
        device = next(self.experts[0].parameters()).device

        # 添加新专家
        with torch.no_grad():
            weights = torch.stack([e.weight.data.clone() for e in self.experts])
            biases = torch.stack([e.bias.data.clone() for e in self.experts])
            
            avg_weight = torch.mean(weights, dim=0).to(device)
            avg_bias = torch.mean(biases, dim=0).to(device)

        for i in range(old_num, new_num_experts):
            new_expert = nn.Linear(input_dim, output_dim).to(device)
            new_expert.weight.data.copy_(avg_weight)
            new_expert.bias.data.copy_(avg_bias)
            
            # 添加噪声
            noise_scale = 0.01
            new_expert.weight.data.add_(torch.randn_like(avg_weight) * noise_scale)
            new_expert.bias.data.add_(torch.randn_like(avg_bias) * noise_scale)
            
            self.experts.append(new_expert)

        # 扩展门控网络
        if self._old_gate_weights is None:
            self._old_gate_weights = []
            for layer in self.gate:
                if isinstance(layer, nn.Linear):
                    self._old_gate_weights.append({
                        'weight': layer.weight.data.clone(),
                        'bias': layer.bias.data.clone()
                    })

        new_gate = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, new_num_experts)
        ).to(device)
        
        with torch.no_grad():
            if self._old_gate_weights is not None:
                new_gate[0].weight.data.copy_(self._old_gate_weights[0]['weight'])
                new_gate[0].bias.data.copy_(self._old_gate_weights[0]['bias'])
                
                new_gate[3].weight.data.copy_(self._old_gate_weights[1]['weight'])
                new_gate[3].bias.data.copy_(self._old_gate_weights[1]['bias'])
                
                old_output_dim = self._old_gate_weights[2]['weight'].size(0)
                new_gate[5].weight.data[:old_output_dim] = self._old_gate_weights[2]['weight']
                new_gate[5].bias.data[:old_output_dim] = self._old_gate_weights[2]['bias']
                
                if new_num_experts > old_num:
                    new_gate[5].weight[old_output_dim:].normal_(mean=-0.1, std=0.01)
                    new_gate[5].bias[old_output_dim:].fill_(-0.5)
        
        self.gate = new_gate
        self.num_experts = new_num_experts
        
        print(f"✅ MoE expanded to {new_num_experts} experts.")
        
        # 重置统计（如果启用了统计记录）
        if self.print_prob > 0:
            self.routing_acc_history = []
            self.expert_usage_history = []

    def get_stats(self):
        """获取统计信息（仅当启用了统计记录时有效）"""
        if self.print_prob <= 0 or not self.routing_acc_history:
            return {"message": "Statistics recording is disabled (print_prob=0)"}
        
        avg_acc = np.mean(self.routing_acc_history)
        
        if self.expert_usage_history and len(self.expert_usage_history) > 0:
            avg_usage = np.mean(self.expert_usage_history, axis=0)
            min_usage = avg_usage.min()
            max_usage = avg_usage.max()
            imbalance = max_usage / (min_usage + 1e-6)
        else:
            avg_usage = np.zeros(self.num_experts)
            imbalance = 0
        
        return {
            "average_routing_accuracy": avg_acc,
            "expert_usage": avg_usage,
            "imbalance_ratio": imbalance,
            "total_samples": len(self.routing_acc_history)
        }