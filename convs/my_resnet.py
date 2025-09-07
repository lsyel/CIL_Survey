"""ResNet in PyTorch with MoE Support.
BasicBlock and Bottleneck module is from the original ResNet paper:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
PreActBlock and PreActBottleneck module is from the later paper:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
Original code is from https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
Modified to support MoE layer for incremental learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBottleneck(nn.Module):
    """Pre-activation version of the original Bottleneck module."""

    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


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


# ============= ResNet with MoE =============
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, use_moe=True, moe_experts=1):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.use_moe = use_moe

        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.out_dim = 512 * block.expansion

        self.moe_layer = None
        if use_moe:
            self.moe_layer = MoELayer(
                input_dim=self.out_dim,
                expert_dim=self.out_dim,
                num_experts=moe_experts,
                k=1
            )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, task_id=None):
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        pooled = F.avg_pool2d(out4, 4)
        features = pooled.view(pooled.size(0), -1)  # (B, D)

        # 👇 插入 MoE 层
        if self.use_moe and self.moe_layer is not None:
            features = self.moe_layer(features, task_id=task_id)

        # logits = self.linear(features)

        return {
            "fmaps": [out1, out2, out3, out4],
            "features": features,
            # "logits": logits  # 保持与之前一致的输出格式
        }

    # ========== 以下为兼容旧接口的方法 ==========
    def feature_list(self, x):
        """兼容旧代码"""
        out_list = []
        out = F.relu(self.bn1(self.conv1(x)))
        out_list.append(out)
        out = self.layer1(out)
        out_list.append(out)
        out = self.layer2(out)
        out_list.append(out)
        out = self.layer3(out)
        out_list.append(out)
        out = self.layer4(out)
        out_list.append(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return y, out_list

    def intermediate_forward(self, x, layer_index):
        out = F.relu(self.bn1(self.conv1(x)))
        if layer_index == 1:
            out = self.layer1(out)
        elif layer_index == 2:
            out = self.layer1(out)
            out = self.layer2(out)
        elif layer_index == 3:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
        elif layer_index == 4:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
        return out

    def penultimate_forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        penultimate = self.layer4(out)
        out = F.avg_pool2d(penultimate, 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return y, penultimate


# ============= Model Builders =============
def ResNet18(num_c, use_moe=False, moe_experts=1):
    return ResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_c, use_moe=use_moe, moe_experts=moe_experts)


def ResNet34(num_c, use_moe=False, moe_experts=1):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_c, use_moe=use_moe, moe_experts=moe_experts)


def ResNet50(num_c, use_moe=False, moe_experts=1):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_c, use_moe=use_moe, moe_experts=moe_experts)


def ResNet101(num_c, use_moe=False, moe_experts=1):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_c, use_moe=use_moe, moe_experts=moe_experts)


def ResNet152(num_c, use_moe=False, moe_experts=1):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_c, use_moe=use_moe, moe_experts=moe_experts)