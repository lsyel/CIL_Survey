import torch
import torch.nn as nn
import torch.nn.functional as F

# ============= MoE Layer =============
class MoELayer(nn.Module):
    def __init__(self, input_dim, expert_dim, num_experts, k=1):
        """
        Simple MoE Layer with Softmax Gating
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
        self.gate = nn.Linear(input_dim, num_experts)

        # 保存旧门控权重，用于扩展时初始化
        self._old_gate_weight = None
        self._old_gate_bias = None

    def forward(self, x, task_id=None):
        """
        x: (B, D)
        task_id: int, 可选。若提供，则强制路由到指定专家（用于增量学习控制）
        """
        B, D = x.shape
        gate_logits = self.gate(x)  # (B, E)

        if task_id is not None:
            # 创建掩码，只允许当前任务专家被选中
            mask = torch.full_like(gate_logits, float('-inf'))
            mask[:, task_id] = 0
            gate_logits = gate_logits + mask

        topk_vals, topk_idxs = torch.topk(gate_logits, self.k, dim=1)  # (B, k)
        topk_vals = F.softmax(topk_vals, dim=1)  # (B, k)

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

        return out

    def expand_experts(self, new_num_experts):
        """动态增加专家数量"""
        if new_num_experts <= self.num_experts:
            return

        old_num = self.num_experts
        input_dim = self.experts[0].in_features
        output_dim = self.experts[0].out_features

        # 添加新专家
        for i in range(old_num, new_num_experts):
            new_expert = nn.Linear(input_dim, output_dim)
            # 可选：用Xavier初始化新专家
            nn.init.xavier_uniform_(new_expert.weight)
            nn.init.zeros_(new_expert.bias)
            self.experts.append(new_expert)

        # 保存当前门控权重
        if self._old_gate_weight is None:
            self._old_gate_weight = self.gate.weight.data.clone()
            self._old_gate_bias = self.gate.bias.data.clone()

        # 扩展门控层
        new_gate = nn.Linear(self.gate.in_features, new_num_experts)
        with torch.no_grad():
            new_gate.weight[:old_num, :] = self._old_gate_weight
            new_gate.bias[:old_num] = self._old_gate_bias
            # 新专家门控初始化为0（偏向不激活）
            nn.init.zeros_(new_gate.weight[old_num:, :])
            nn.init.zeros_(new_gate.bias[old_num:])

        self.gate = new_gate
        self.num_experts = new_num_experts

        print(f"✅ MoE expanded to {new_num_experts} experts.")