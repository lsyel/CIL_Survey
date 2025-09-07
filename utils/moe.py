# models/moe.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
    def __init__(self, input_dim, expert_dim, num_experts, k=1):
        """
        :param input_dim: 输入特征维度
        :param expert_dim: 每个专家输出维度（通常=input_dim）
        :param num_experts: 当前专家数量
        :param k: top-k 路由，通常=1
        """
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.k = k
        self.experts = nn.ModuleList([
            nn.Linear(input_dim, expert_dim) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(input_dim, num_experts)  # 可学习路由门控

    def forward(self, x, task_id=None):
        """
        x: (B, D)
        task_id: 可选，用于在增量时强制路由到新专家（调试/控制用）
        """
        B, D = x.shape
        gate_logits = self.gate(x)  # (B, num_experts)
        
        if task_id is not None:
            # 强制路由到特定专家（如当前任务对应的专家）
            # 假设每个任务对应一个专家，task_id 从0开始
            mask = torch.full_like(gate_logits, float('-inf'))
            mask[:, task_id] = 0
            gate_logits = gate_logits + mask

        # Top-k 路由
        topk_vals, topk_idxs = torch.topk(gate_logits, self.k, dim=1)  # (B, k)
        topk_vals = F.softmax(topk_vals, dim=1)  # (B, k)

        out = torch.zeros(B, self.experts[0].out_features, device=x.device)
        
        # 对每个专家，聚合分配给它的样本
        for i in range(self.num_experts):
            mask = (topk_idxs == i).any(dim=1)  # (B,)
            if mask.any():
                expert_out = self.experts[i](x[mask])  # (n, D)
                # 获取对应权重
                weights = topk_vals[mask]  # (n, k)
                idx_in_topk = (topk_idxs[mask] == i).float()  # (n, k)
                weighted_weights = (weights * idx_in_topk).sum(dim=1, keepdim=True)  # (n, 1)
                out[mask] += weighted_weights * expert_out

        return out

    def expand_experts(self, new_num_experts):
        """动态增加专家"""
        old_num = len(self.experts)
        if new_num_experts <= old_num:
            return
        for i in range(old_num, new_num_experts):
            self.experts.append(
                nn.Linear(self.experts[0].in_features, self.experts[0].out_features)
            )
        # 重新初始化门控层
        self.gate = nn.Linear(self.gate.in_features, new_num_experts)
        # 可选：保留旧门控权重
        with torch.no_grad():
            if hasattr(self, '_old_gate_weight'):
                self.gate.weight[:old_num, :] = self._old_gate_weight
                self.gate.bias[:old_num] = self._old_gate_bias
            else:
                self._old_gate_weight = self.gate.weight.clone()
                self._old_gate_bias = self.gate.bias.clone()