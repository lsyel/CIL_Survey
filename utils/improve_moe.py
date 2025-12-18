import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy

# ============= 修复的版本1：单隐藏层增强专家 =============
class EnhancedExpertV1(nn.Module):
    """修复版本：单隐藏层增强专家网络，包含正确的初始化"""
    def __init__(self, input_dim, expert_dim, hidden_ratio=2.0, dropout=0.1, activation='relu'):
        super(EnhancedExpertV1, self).__init__()
        
        # 隐藏层维度
        hidden_dim = int(expert_dim * hidden_ratio)
        
        # 网络结构：输入 -> 隐藏层 -> 输出
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True) if activation == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, expert_dim)
        )
        
        # 初始化权重
        self._init_weights()
        
        # 为了保持兼容性
        self._in_features = input_dim
        self._out_features = expert_dim
        
        # 保存结构信息用于调试
        self.hidden_dim = hidden_dim
        self.dropout = dropout
    
    def _init_weights(self):
        """初始化权重，避免梯度爆炸/消失"""
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                # 使用Xavier初始化，对线性层更稳定
                nn.init.xavier_uniform_(layer.weight, gain=1.0)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        # 添加数值稳定性检查
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
            
        out = self.net(x)
        
        # 检查输出
        if torch.isnan(out).any() or torch.isinf(out).any():
            out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
            
        return out
    
    # 为了兼容原来的代码
    @property
    def in_features(self):
        return self._in_features
    
    @property
    def out_features(self):
        return self._out_features
    
    @property
    def weight(self):
        return self.net[0].weight
    
    @property
    def bias(self):
        return self.net[0].bias


# ============= 修复的MoELayer =============
class EnhancedMoELayer(nn.Module):
    def __init__(self, input_dim, expert_dim, num_experts, k=1,
                 distill_weight=1, temperature=2.0,
                 expert_type='simple',  # 'simple' 或 'enhanced_v1'
                 hidden_ratio=2.0,      # 增强专家的隐藏层比例
                 expert_dropout=0.1,    # 增强专家的dropout
                 expert_activation='relu'):  # 激活函数
        super(EnhancedMoELayer, self).__init__()
        self.num_experts = num_experts
        self.k = k
        self.distill_weight = distill_weight
        self.temperature = temperature
        self.expert_type = expert_type
        self.hidden_ratio = hidden_ratio
        self.expert_dropout = expert_dropout
        self.expert_activation = expert_activation
        
        # 专家网络
        self.experts = self._build_experts(input_dim, expert_dim, num_experts)
        
        # 门控网络
        self.gate = self._build_gate_network(input_dim, num_experts)
        
        # 初始化门控网络
        self._init_gate_weights()
        
        # 旧门控网络
        self.old_gate = self.gate
        self.old_num_experts = 0
        self._old_gate_weights = None
        
        print(f"✅ Built {expert_type} MoE with {num_experts} experts")
        if expert_type == 'enhanced_v1':
            print(f"   Hidden ratio: {hidden_ratio}, Dropout: {expert_dropout}")

    def _build_experts(self, input_dim, expert_dim, num_experts):
        """构建专家网络"""
        experts = nn.ModuleList()
        
        for _ in range(num_experts):
            if self.expert_type == 'simple':
                expert = nn.Linear(input_dim, expert_dim)
                # 初始化简单专家
                nn.init.xavier_uniform_(expert.weight, gain=1.0)
                nn.init.constant_(expert.bias, 0)
            elif self.expert_type == 'enhanced_v1':
                expert = EnhancedExpertV1(
                    input_dim=input_dim,
                    expert_dim=expert_dim,
                    hidden_ratio=self.hidden_ratio,
                    dropout=self.expert_dropout,
                    activation=self.expert_activation
                )
            else:
                raise ValueError(f"Unknown expert type: {self.expert_type}")
            experts.append(expert)
        
        return experts

    def _build_gate_network(self, input_dim, num_experts):
        """构建门控网络"""
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

    def _init_gate_weights(self):
        """初始化门控网络权重"""
        for layer in self.gate:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0)

    def set_old_gate(self, old_gate, old_num_experts):
        """设置旧门控网络用于蒸馏"""
        if old_gate is not None:
            self.old_gate = copy.deepcopy(old_gate)
            self.old_gate.eval()
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
        
        # 检查输入
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        # 门控logits
        gate_logits = self.gate(x)
        
        # 检查门控输出
        if torch.isnan(gate_logits).any():
            gate_logits = torch.nan_to_num(gate_logits, nan=0.0)
        
        # 计算路由损失
        routing_loss = 0
        distill_loss = 0
        total_routing_loss = 0
        
        if routing_targets is not None and self.training:
            # 基础路由损失
            routing_loss = F.cross_entropy(gate_logits, routing_targets)
            
            # 计算蒸馏损失
            if self.old_gate is not None and self.old_num_experts > 0:
                distill_loss = self._compute_distill_loss(x, gate_logits)
                total_routing_loss = routing_loss + self.distill_weight * distill_loss
                if random.random() < 0.1:
                    print(f"total_routing_loss: {total_routing_loss.item():.4f}, "
                          f"routing_loss: {routing_loss.item():.4f}, "
                          f"distill_loss: {distill_loss.item():.4f}")
            else:
                total_routing_loss = routing_loss
        
        # Top-k 选择
        topk_vals, topk_idxs = torch.topk(gate_logits, self.k, dim=1)
        topk_vals = F.softmax(topk_vals, dim=1)
        
        # 计算输出
        if self.expert_type == 'simple':
            out_features = self.experts[0].out_features
        else:
            out_features = self.experts[0]._out_features
        
        out = torch.zeros(B, out_features, device=x.device)
        
        for i in range(self.num_experts):
            expert_mask = (topk_idxs == i).any(dim=1)
            if expert_mask.any():
                batch_x = x[expert_mask]
                expert_out = self.experts[i](batch_x)
                
                # 检查专家输出
                if torch.isnan(expert_out).any():
                    expert_out = torch.nan_to_num(expert_out, nan=0.0)
                
                weights = topk_vals[expert_mask]
                idx_match = (topk_idxs[expert_mask] == i).float()
                weighted = (weights * idx_match).sum(dim=1, keepdim=True)
                
                out[expert_mask] += weighted * expert_out
        
        # 检查最终输出
        if torch.isnan(out).any():
            out = torch.nan_to_num(out, nan=0.0)
        
        return {
            "output": out,
            "routing_loss": total_routing_loss,
            "gate_logits": gate_logits,
            "expert_assignments": topk_idxs
        }

    def _compute_distill_loss(self, x, current_gate_logits):
        """计算门控网络蒸馏损失"""
        with torch.no_grad():
            old_gate_logits = self.old_gate(x)
            
            if self.old_num_experts < self.num_experts:
                expanded_old_logits = torch.zeros_like(current_gate_logits)
                expanded_old_logits[:, :self.old_num_experts] = old_gate_logits
                old_gate_logits = expanded_old_logits
            elif self.old_num_experts > self.num_experts:
                old_gate_logits = old_gate_logits[:, :self.num_experts]
        
        T = self.temperature
        current_probs = F.log_softmax(current_gate_logits / T, dim=1)
        old_probs = F.softmax(old_gate_logits / T, dim=1)
        
        distill_loss = F.kl_div(current_probs, old_probs, reduction='batchmean') * (T ** 2)
        
        # 检查损失是否为NaN
        if torch.isnan(distill_loss):
            distill_loss = torch.tensor(0.0, device=distill_loss.device)
        
        return distill_loss

    def expand_experts(self, new_num_experts):
        """扩展专家"""
        if new_num_experts <= self.num_experts:
            return

        old_num = self.num_experts
        self._freeze_experts(old_num)

        # 保存当前门控网络作为旧门控网络
        self.set_old_gate(self.gate, old_num)
        
        # 获取网络参数
        if self.expert_type == 'simple':
            input_dim = self.experts[0].in_features
            output_dim = self.experts[0].out_features
        else:
            input_dim = self.experts[0]._in_features
            output_dim = self.experts[0]._out_features
            
        device = next(self.experts[0].parameters()).device

        # 添加新专家
        self._add_new_experts(old_num, new_num_experts, input_dim, output_dim, device)
        
        # 扩展门控网络
        self.gate = self._expand_gate_network(new_num_experts, device)
        self.num_experts = new_num_experts
        
        print(f"✅ {self.expert_type} MoE expanded to {new_num_experts} experts.")
        self.distill_weight += 0.0

    def _add_new_experts(self, old_num, new_num_experts, input_dim, output_dim, device):
        """添加新专家 - 修复版本"""
        # 计算所有旧专家的平均值
        with torch.no_grad():
            if self.expert_type == 'simple':
                weights = torch.stack([e.weight.data.clone() for e in self.experts])
                biases = torch.stack([e.bias.data.clone() for e in self.experts])
            else:
                # 对于增强专家，只平均第一层权重
                weights = torch.stack([e.net[0].weight.data.clone() for e in self.experts])
                biases = torch.stack([e.net[0].bias.data.clone() for e in self.experts])
            
            avg_weight = torch.mean(weights, dim=0)
            avg_bias = torch.mean(biases, dim=0)

        # 添加新专家
        for i in range(old_num, new_num_experts):
            if self.expert_type == 'simple':
                new_expert = nn.Linear(input_dim, output_dim).to(device)
                new_expert.weight.data.copy_(avg_weight)
                new_expert.bias.data.copy_(avg_bias)
                
                # 添加噪声
                noise_scale = 0.01
                new_expert.weight.data.add_(torch.randn_like(avg_weight) * noise_scale)
                new_expert.bias.data.add_(torch.randn_like(avg_bias) * noise_scale)
                
            else:  # enhanced_v1
                new_expert = EnhancedExpertV1(
                    input_dim=input_dim,
                    expert_dim=output_dim,
                    hidden_ratio=self.hidden_ratio,
                    dropout=self.expert_dropout,
                    activation=self.expert_activation
                ).to(device)
                
                # 使用平均权重初始化第一层
                with torch.no_grad():
                    new_expert.net[0].weight.data.copy_(avg_weight)
                    new_expert.net[0].bias.data.copy_(avg_bias)
                    
                    # 为第一层添加噪声
                    noise_scale = 0.01
                    new_expert.net[0].weight.data.add_(torch.randn_like(avg_weight) * noise_scale)
                    new_expert.net[0].bias.data.add_(torch.randn_like(avg_bias) * noise_scale)
                    
                    # 注意：我们不需要手动初始化第二层，因为 EnhancedExpertV1.__init__ 已经初始化了
                    # 但我们可以检查一下第二层是否被正确初始化
                    if len(new_expert.net) > 3:
                        second_layer = new_expert.net[3]
                        if isinstance(second_layer, nn.Linear):
                            # 确保第二层是正确初始化的
                            if torch.all(second_layer.weight == 0):
                                nn.init.xavier_uniform_(second_layer.weight, gain=1.0)
                            if second_layer.bias is not None and torch.all(second_layer.bias == 0):
                                nn.init.constant_(second_layer.bias, 0)
            
            self.experts.append(new_expert)

    def _expand_gate_network(self, new_num_experts, device):
        """扩展门控网络"""
        if self._old_gate_weights is None:
            self._old_gate_weights = []
            for layer in self.gate:
                if isinstance(layer, nn.Linear):
                    self._old_gate_weights.append({
                        'weight': layer.weight.data.clone(),
                        'bias': layer.bias.data.clone()
                    })

        # 构建新门控网络
        if self.expert_type == 'simple':
            in_features = self.experts[0].in_features
        else:
            in_features = self.experts[0]._in_features
            
        new_gate = self._build_gate_network(in_features, new_num_experts).to(device)
        
        # 初始化新门控网络
        for layer in new_gate:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0)
        
        # 复制旧权重到新门控网络
        self._copy_gate_weights(new_gate, new_num_experts)
        
        return new_gate

    def _copy_gate_weights(self, new_gate, new_num_experts):
        """复制旧门控网络的权重到新门控网络"""
        if self._old_gate_weights is None:
            return
            
        with torch.no_grad():
            new_gate[0].weight.data.copy_(self._old_gate_weights[0]['weight'])
            new_gate[0].bias.data.copy_(self._old_gate_weights[0]['bias'])
            
            new_gate[3].weight.data.copy_(self._old_gate_weights[1]['weight'])
            new_gate[3].bias.data.copy_(self._old_gate_weights[1]['bias'])
            
            if len(self._old_gate_weights) > 4:
                new_gate[6].weight.data.copy_(self._old_gate_weights[4]['weight'])
                new_gate[6].bias.data.copy_(self._old_gate_weights[4]['bias'])
            
            old_output_dim = self._old_gate_weights[-1]['weight'].size(0)
            new_gate[-1].weight.data[:old_output_dim] = self._old_gate_weights[-1]['weight']
            new_gate[-1].bias.data[:old_output_dim] = self._old_gate_weights[-1]['bias']
            
            if new_num_experts > old_output_dim:
                avg_weight = torch.mean(self._old_gate_weights[-1]['weight'], dim=0)
                avg_bias = torch.mean(self._old_gate_weights[-1]['bias'], dim=0)
                
                for i in range(old_output_dim, new_num_experts):
                    new_gate[-1].weight.data[i] = avg_weight.clone()
                    new_gate[-1].bias.data[i] = avg_bias.clone()
                    
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