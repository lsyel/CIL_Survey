import logging
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import os
from models.base import BaseLearner
from utils.inc_net import IncrementalNet  # 确保它支持 use_moe 和 update_moe_experts
from utils.toolkit import target2onehot, tensor2numpy
import random
EPSILON = 1e-8

# ========== 超参数 ==========
init_epoch = 50
init_lr = 0.1
init_milestones = [60, 120, 170]
init_lr_decay = 0.1
init_weight_decay = 0.0005

epochs = 20
lrate = 0.1
milestones = [80, 120]
lrate_decay = 0.1
batch_size = 128
weight_decay = 2e-4
num_workers = 8
T = 2  # 蒸馏温度


class iCaRLMoe(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        # 👇 初始化网络时传入 use_moe 参数
        self._network = IncrementalNet(
            args["convnet_type"],
            False,
            use_moe=True
        )
        self._cur_task = -1  # 初始化为 -1，第一个任务变成 0
    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))
        self._save_model()
    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)

        # 👇 新增：扩展 MoE 专家数量（每个任务一个专家）
        if hasattr(self._network, 'update_moe_experts'):
            self._network.update_moe_experts(self._cur_task)

        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

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

        # 是否跳过初始训练（用于加载预训练模型）
        if self.args['skip'] and self._cur_task == 0:
            load_acc = self._network.load_checkpoint(self.args)

        # 多 GPU 支持
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        # 开始训练
        if self._cur_task == 0:
            if self.args['skip']:
                self._network.to(self._device)
                cur_test_acc = self._compute_accuracy(self._network, self.test_loader)
                logging.info(f"Loaded_Test_Acc:{load_acc} Cur_Test_Acc:{cur_test_acc}")
            else:
                self._train(self.train_loader, self.test_loader)
        else:
            self._train(self.train_loader, self.test_loader)

        # 构建回放记忆
        self.build_rehearsal_memory(data_manager, self.samples_per_class)

        # 如果用了 DataParallel，恢复为单模块
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
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                # 👇 传入 task_id 控制 MoE 路由
                output = self._network(inputs, task_id=None)
                logits = output["logits"]

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

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
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
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
                routing_loss_weight = 0.1
                
                # ===== 反向传播 =====
                optimizer.zero_grad()
                
                # 1. 先计算主损失的梯度
                main_loss.backward(retain_graph=True)  # 保留计算图以便后续计算路由损失
                
                # 2. 再计算路由损失的梯度
                weighted_routing_loss = routing_loss_weight * routing_loss
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
            prog_bar.set_description(info)
            logging.info(info)

    def eval_task(self, save_conf=False):
        """评估模型性能，并计算每个类别的准确率"""
        cnn_pred_list, cnn_target_list, cnn_logits_list = [], [], []
        self._network.eval()

        # 初始化类别统计
        class_correct = [0] * self._total_classes
        class_total = [0] * self._total_classes
        
        # 使用进度条显示评估过程
        progress_bar = tqdm(self.test_loader, desc="评估任务")
        for i, (_, inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs, task_id=None)
                logits = outputs["logits"]
                cnn_logits_list.append(logits.cpu().numpy())
            
            cnn_preds = torch.max(logits, dim=1)[1]
            cnn_pred_list.append(cnn_preds.cpu().numpy())
            cnn_target_list.append(targets.cpu().numpy())
            
            # 统计每个类别的正确预测数
            for t, p in zip(targets.cpu().numpy(), cnn_preds.cpu().numpy()):
                if t < self._total_classes:  # 确保类别索引有效
                    class_total[t] += 1
                    if t == p:
                        class_correct[t] += 1

        cnn_pred_all = np.concatenate(cnn_pred_list)
        cnn_target_all = np.concatenate(cnn_target_list)
        cnn_logits_all = np.vstack(cnn_logits_list)

        # 计算整体准确率
        cnn_accy = self._evaluate(cnn_pred_all, cnn_target_all, cnn_logits_all, self._total_classes)
        
        # 计算并打印每个类别的准确率
        logging.info("\n类别准确率:")
        logging.info("=" * 50)
        logging.info(f"{'类别':<15} | {'样本数':<8} | {'正确数':<8} | {'准确率':<8}")
        logging.info("-" * 50)
        
        # 获取类别标签映射（如果有）
        if hasattr(self, 'data_manager') and hasattr(self.data_manager, 'class_order'):
            class_labels = self.data_manager.class_order
        else:
            class_labels = [str(i) for i in range(self._total_classes)]
        
        # 计算平均准确率
        total_acc = 0.0
        valid_classes = 0
        
        for i in range(self._total_classes):
            if class_total[i] > 0:
                acc = 100 * class_correct[i] / class_total[i]
                total_acc += acc
                valid_classes += 1
            else:
                acc = 0.0
            logging.info(f"{class_labels[i]:<15} | {class_total[i]:<8} | {class_correct[i]:<8} | {acc:.2f}%")
        
        # 计算平均准确率
        if valid_classes > 0:
            avg_acc = total_acc / valid_classes
            logging.info("-" * 50)
            logging.info(f"{'平均准确率':<15} | {'':<8} | {'':<8} | {avg_acc:.2f}%")
        
        logging.info("=" * 50)

        nme_accy = None

        if save_conf:
            # 保存混淆矩阵供后续分析
            confusion = confusion_matrix(cnn_target_all, cnn_pred_all)
            np.save(os.path.join(self.args["logfilename"], f"confusion_task_{self._cur_task}.npy"), confusion)
            
            # 保存预测结果
            np.save(os.path.join(self.args["logfilename"], "cnn_pred.npy"), cnn_pred_all)
            np.save(os.path.join(self.args["logfilename"], "cnn_target.npy"), cnn_target_all)
            np.save(os.path.join(self.args["logfilename"], "cnn_logits.npy"), cnn_logits_all)

        return cnn_accy, nme_accy

    def _compute_accuracy(self, model, loader):
        """
        辅助函数：计算模型在数据加载器上的准确率
        """
        model.eval()
        correct, total = 0, 0
        device = next(model.parameters()).device  # 自动获取模型设备

        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            with torch.no_grad():
                # 👇 评估时传入当前任务 ID（也可设为 None）
                outputs = model(inputs, task_id=None)
                logits = outputs["logits"]
            predicts = torch.max(logits, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    # 👇 必须缩进在 class iCaRL 下面！
    def _evaluate(self, y_pred, y_true, y_logits, total_classes):
        """
        使用 logits 计算 top1, top3
        """
        ret = {}

        # ===== Top-1 =====
        ret["top1"] = (y_pred == y_true).sum() / len(y_true)

        # ===== Top-3 =====
        top3_correct = 0
        for i in range(len(y_true)):
            # 获取 logits 排名前三的类别
            top3 = np.argsort(y_logits[i])[-3:][::-1]  # 降序取前3
            if y_true[i] in top3:
                top3_correct += 1
        ret["top3"] = top3_correct / len(y_true)

        # ===== Grouped =====
        grouped = {}
        task_size = self.args["increment"]
        for i in range(0, total_classes, task_size):
            mask = (y_true >= i) & (y_true < i + task_size)
            if mask.any():
                grouped[f"{i:0>2d}-{i+task_size-1:0>2d}"] = (y_pred[mask] == y_true[mask]).mean()
        ret["grouped"] = grouped

        return ret
    def _save_model(self):
        model_path = os.path.join("./pth", f"task_{self._cur_task}_model.pth")
        
        # 收集所有参数信息
        param_info = []
        for name, param in self._network.named_parameters():
            param_info.append({
                'name': name,
                'shape': tuple(param.shape),
                'dtype': str(param.dtype),
                'mean': param.mean().item(),
                'std': param.std().item(),
                'min': param.min().item(),
                'max': param.max().item()
            })
        
        # 保存模型状态
        model_state = {
            'network_state_dict': self._network.state_dict(),
            'moe_layer_state': self._network.convnet.moe_layer.state_dict(),
            'total_classes': self._total_classes,
            'known_classes': self._known_classes,
            'cur_task': self._cur_task,
            'data_memory': self._data_memory,
            'targets_memory': self._targets_memory,
            'args': self.args,
            'param_info': param_info,  # 添加参数信息
            'moe_experts': self._network.convnet.moe_layer.num_experts  # 新增
        }
        
        torch.save(model_state, model_path)
        
        # 打印参数摘要
        # self._log_param_summary(param_info, "保存模型参数")
        
        logging.info(f"模型已保存到 {model_path}")
        return model_path

    def _log_param_summary(self, param_info, title):
        """记录参数摘要信息"""
        logging.info(f"\n{'='*50}")
        logging.info(f"{title} - 参数摘要")
        logging.info(f"{'参数名称':<40} | {'形状':<20} | {'均值':<10} | {'标准差':<10} | {'最小值':<10} | {'最大值':<10}")
        logging.info(f"{'-'*100}")
        
        for info in param_info:
            logging.info(
                f"{info['name']:<40} | {str(info['shape']):<20} | "
                f"{info['mean']:>10.6f} | {info['std']:>10.6f} | "
                f"{info['min']:>10.6f} | {info['max']:>10.6f}"
            )
        
        # 添加统计信息
        total_params = sum(np.prod(info['shape']) for info in param_info)
        logging.info(f"\n总计参数数量: {total_params}")
        logging.info(f"{'='*50}\n")

# ========== 辅助函数：知识蒸馏损失 ==========
def _KD_loss(pred, soft, T):
    """
    计算知识蒸馏损失
    :param pred: 当前模型对旧类别的 logits
    :param soft: 旧模型的 logits
    :param T: 温度
    """
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]