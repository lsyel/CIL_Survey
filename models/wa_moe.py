import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import random
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import target2onehot, tensor2numpy

EPSILON = 1e-8

# 超参数
init_epoch = 50
init_lr = 0.1
init_milestones = [40]
init_lr_decay = 0.1
init_weight_decay = 0.0005

epochs = 20
lrate = 0.1
milestones = [15]
lrate_decay = 0.1
batch_size = 128
weight_decay = 2e-4
num_workers = 8
T = 2

# MoE相关超参数
routing_loss_weight = 0.1  # 路由损失权重

class WA_MoE(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        # 👇 初始化网络时启用MoE
        self._network = IncrementalNet(args["convnet_type"], False, use_moe=True)
        self._cur_task = -1  # 初始化为-1，第一个任务变成0

    def after_task(self):
        if self._cur_task > 0:
            # 权重对齐（只影响分类层，不影响MoE层）
            if len(self._multiple_gpus) > 1:
                self._network.module.weight_align(self._total_classes - self._known_classes)
            else:
                self._network.weight_align(self._total_classes - self._known_classes)
        
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        
        # 👇 新增：扩展MoE专家数量（每个任务一个专家）
        if hasattr(self._network, 'update_moe_experts'):
            self._network.update_moe_experts(self._cur_task)
        
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        # 获取任务大小（每个任务的类别数）
        task_size = self.args["increment"]

        # 构建训练数据集（当前任务 + 回放样本）
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        
        # 构建测试数据集（所有已见类别）
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        # 多GPU支持
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        
        # 开始训练
        self._train(self.train_loader, self.test_loader)
        
        # 构建回放记忆
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        
        # 恢复为单模块
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        # 根据是否是初始任务设置优化器和调度器
        if self._cur_task == 0:
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=init_lr,
                weight_decay=init_weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay
            )
            if self.args['skip']:
                if len(self._multiple_gpus) > 1:
                    self._network = self._network.module
                load_acc = self._network.load_checkpoint(self.args)
                self._network.to(self._device)
                cur_test_acc = self._compute_accuracy(self._network, self.test_loader)
                logging.info(f"Loaded_Test_Acc:{load_acc} Cur_Test_Acc:{cur_test_acc}")
                if len(self._multiple_gpus) > 1:
                    self._network = nn.DataParallel(self._network, self._multiple_gpus)
            else:
                self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(
                self._network.parameters(),
                lr=lrate,
                momentum=0.9,
                weight_decay=weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=milestones, gamma=lrate_decay
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(init_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            routing_losses = 0.0  # 单独记录路由损失
            correct, total = 0, 0
            
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                
                # ===== 计算路由目标 =====
                # 第一个任务所有样本都路由到专家0
                routing_targets = torch.zeros_like(targets, device=self._device)
                
                # ===== 前向传播 =====
                output = self._network(inputs, task_id=None, routing_targets=routing_targets)
                logits = output["logits"]
                
                # ===== 损失计算 =====
                loss_clf = F.cross_entropy(logits, targets)
                
                # 路由损失权重
                
                # 总损失
                loss = loss_clf 
                
                # ===== 反向传播 =====
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # ===== 记录损失 =====
                losses += loss_clf.item()
                # ===== 统计准确率 =====
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, , Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    routing_losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            routing_losses = 0.0  # 单独记录路由损失
            correct, total = 0, 0
            
            # 获取任务大小（每个任务的类别数）
            task_size = self.args["increment"]
            
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                
                # ===== 计算路由目标 =====
                routing_targets = torch.zeros_like(targets, device=self._device)
                
                # 新样本（当前任务）：应该路由到新专家
                new_sample_mask = (targets >= self._known_classes)
                routing_targets[new_sample_mask] = self._cur_task
                
                # 回放样本（旧样本）：应该路由到旧专家
                replay_mask = (targets < self._known_classes)
                if replay_mask.any() and self._cur_task > 0:
                    # 计算样本的原始任务ID
                    task_origin = targets // task_size
                    
                    # 确保任务ID在有效范围内 [0, self._cur_task-1]
                    task_origin = torch.clamp(task_origin, 0, self._cur_task - 1)
                    
                    routing_targets[replay_mask] = task_origin[replay_mask]
                
                # ===== 前向传播 =====
                output = self._network(inputs, task_id=None, routing_targets=routing_targets)
                logits = output["logits"]
                routing_loss = output.get("routing_loss", 0)  # 获取路由损失
                
                # ===== 损失计算 =====
                # 分类损失
                loss_clf = F.cross_entropy(logits, targets)
                
                # 蒸馏损失（旧类别部分）
                if self._old_network is not None:
                    with torch.no_grad():
                        old_output = self._old_network(inputs, None)
                    old_logits = old_output["logits"][:, :self._known_classes]
                    current_old_logits = logits[:, :self._known_classes]
                    loss_kd = _KD_loss(current_old_logits, old_logits, T)
                else:
                    loss_kd = 0
                
                # 主损失（分类 + 蒸馏）
                main_loss = loss_clf + loss_kd
                
                # 路由损失权重
                weighted_routing_loss = routing_loss_weight * routing_loss
                
                # ===== 反向传播 =====
                optimizer.zero_grad()
                
                # 1. 先计算主损失的梯度
                main_loss.backward(retain_graph=True)  # 保留计算图以便后续计算路由损失
                
                # 2. 再计算路由损失的梯度
                weighted_routing_loss.backward()
                
                # 3. 更新参数
                optimizer.step()
                
                # ===== 记录损失 =====
                losses += main_loss.item()
                routing_losses += weighted_routing_loss.item()
                
                # ===== 统计准确率 =====
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                
                if random.random() < 0.01:
                    logging.info(f"Main loss: {main_loss.item():.4f}, Routing loss: {weighted_routing_loss.item():.4f}")
            
            # ===== 更新学习率 =====
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                # ===== 计算测试准确率 =====
                test_acc = self._compute_accuracy(self._network, test_loader)
                
                # ===== 日志记录 =====
                info = "Task {}, Epoch {}/{} => Main Loss {:.3f}, Routing Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    routing_losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Main Loss {:.3f}, Routing Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    routing_losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
            logging.info(info)
    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
