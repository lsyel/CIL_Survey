import logging
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, accuracy_score
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
import pandas as pd
import json

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
        self.routing_eval_history = []  # 可选：记录路由评估历史
        self.performance_history = []  # 存储每个任务的性能指标历史

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
        super().after_task()

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
        clf_loss_weight = self.args.get("clf_loss_weight", 1.0)+0.0*self._cur_task
        kd_loss_weight = self.args.get("kd_loss_weight", 1.5)+0.5*self._cur_task
        moe_loss_weight = self.args.get("moe_loss_weight", 0.1)+0.05*self._cur_task
        logging.info(f"clf_loss_weight:{clf_loss_weight} kd_loss_weight:{kd_loss_weight} moe_loss_weight:{moe_loss_weight}")
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            routing_correct, routing_total = 0, 0
            
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
                # ===== 新增：计算路由准确率 =====
                with torch.no_grad():
                    gate_logits = output.get("gate_logits")
                    if gate_logits is not None:
                        predicted_tasks = torch.argmax(gate_logits, dim=1)
                        routing_correct += (predicted_tasks == routing_targets).sum().item()
                        routing_total += len(targets)
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
                
                clf_loss = clf_loss_weight * loss_clf
                kd_loss = kd_loss_weight * loss_kd
                moe_loss = moe_loss_weight * routing_loss
                loss = clf_loss + kd_loss + moe_loss
                # ===== 反向传播 =====
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # ===== 记录损失 =====
                losses += loss.item()
                
                # ===== 统计准确率 =====
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                
            # ===== 新增：打印路由准确率 =====
            if routing_total > 0:
                routing_accuracy = 100.0 * routing_correct / routing_total
                logging.info(f"路由准确率: {routing_accuracy:.2f}% ({routing_correct}/{routing_total})")
            # ===== 更新学习率 =====
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                # ===== 计算测试准确率 =====
                test_acc = self._compute_accuracy(self._network, test_loader)
                
                # ===== 日志记录 =====
                info = "Task {}, Epoch {}/{} => Main Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Main Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
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
        
        # ===== 新增：在分类评估前先评估路由网络 =====
        if self._cur_task > 0:  # 只有多任务时才评估路由
            logging.info("开始路由网络评估...")
            routing_stats = self.evaluate_routing_network(self.test_loader, phase="test")
            
            # 可选：基于路由性能进行分析
            if routing_stats and routing_stats['overall_accuracy'] < 0.8:
                logging.warning("路由性能较低，可能影响MoE效果")
        
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
        
        # ===== 新增：计算并保存详细性能指标 =====
        detailed_metrics = self._compute_detailed_metrics(cnn_pred_all, cnn_target_all, self._total_classes)
        
        # 将详细指标整合到cnn_accy中
        cnn_accy.update(detailed_metrics)
        
        # 保存当前任务的性能指标
        self._save_task_performance(cnn_accy, cnn_pred_all, cnn_target_all, self._cur_task)
        
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
    
    def _compute_detailed_accuracy(self, model, loader):
        """
        计算详细的性能指标：准确率、精确率、召回率、F1分数
        """
        model.eval()
        all_preds = []
        all_targets = []
        
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs, task_id=None)
                logits = outputs["logits"]
            predicts = torch.max(logits, dim=1)[1]
            all_preds.extend(predicts.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # 计算各种指标
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        
        return {
            'accuracy': accuracy * 100,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1 * 100
        }

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
    
    def _compute_detailed_metrics(self, y_pred, y_true, total_classes):
        """
        计算详细的性能指标：精确率、召回率、F1分数等
        """
        ret = {}
        
        # 计算整体指标（加权平均）
        ret['overall_accuracy'] = accuracy_score(y_true, y_pred)
        ret['overall_precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        ret['overall_recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        ret['overall_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # 计算宏平均指标
        ret['macro_precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        ret['macro_recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        ret['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # 计算微平均指标
        ret['micro_precision'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
        ret['micro_recall'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
        ret['micro_f1'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        
        # 生成完整的分类报告
        class_report = classification_report(
            y_true, y_pred, 
            output_dict=True,
            zero_division=0
        )
        ret['classification_report'] = class_report
        
        # 计算每个类别的指标
        per_class_metrics = {}
        for class_idx in range(total_classes):
            class_mask = y_true == class_idx
            if np.sum(class_mask) > 0:  # 只计算有样本的类别
                per_class_metrics[class_idx] = {
                    'precision': precision_score(y_true, y_pred, average=None, zero_division=0)[class_idx],
                    'recall': recall_score(y_true, y_pred, average=None, zero_division=0)[class_idx],
                    'f1': f1_score(y_true, y_pred, average=None, zero_division=0)[class_idx],
                    'support': int(np.sum(class_mask))
                }
        
        ret['per_class_metrics'] = per_class_metrics
        
        # 打印整体指标
        self._print_detailed_metrics(ret)
        
        return ret
    
    def _print_detailed_metrics(self, metrics):
        """打印详细的性能指标"""
        logging.info("\n📊 详细性能指标:")
        logging.info("=" * 60)
        logging.info(f"{'指标':<20} | {'值':<10}")
        logging.info("-" * 60)
        logging.info(f"{'整体准确率':<20} | {metrics['overall_accuracy']:.4f}")
        logging.info(f"{'整体精确率':<20} | {metrics['overall_precision']:.4f}")
        logging.info(f"{'整体召回率':<20} | {metrics['overall_recall']:.4f}")
        logging.info(f"{'整体F1分数':<20} | {metrics['overall_f1']:.4f}")
        logging.info("-" * 60)
        logging.info(f"{'宏平均精确率':<20} | {metrics['macro_precision']:.4f}")
        logging.info(f"{'宏平均召回率':<20} | {metrics['macro_recall']:.4f}")
        logging.info(f"{'宏平均F1分数':<20} | {metrics['macro_f1']:.4f}")
        logging.info(f"{'微平均精确率':<20} | {metrics['micro_precision']:.4f}")
        logging.info(f"{'微平均召回率':<20} | {metrics['micro_recall']:.4f}")
        logging.info(f"{'微平均F1分数':<20} | {metrics['micro_f1']:.4f}")
        logging.info("=" * 60)
        
        # 打印分类报告摘要
        if 'classification_report' in metrics:
            report = metrics['classification_report']
            logging.info("\n📋 分类报告摘要:")
            logging.info("-" * 60)
            for key, value in report.items():
                if key not in ['accuracy', 'macro avg', 'weighted avg'] and isinstance(value, dict):
                    logging.info(f"类别 {key}: 精确率={value['precision']:.4f}, 召回率={value['recall']:.4f}, F1={value['f1-score']:.4f}, 支持数={value['support']}")
    
    def _save_task_performance(self, metrics, y_pred, y_true, task_id):
        """保存当前任务的性能指标"""
        # 创建结果字典
        task_result = {
            'task_id': task_id,
            'total_classes': self._total_classes,
            'known_classes': self._known_classes,
            'timestamp': pd.Timestamp.now().isoformat(),
            'metrics': {
                'top1_accuracy': float(metrics.get('top1', 0)),
                'top3_accuracy': float(metrics.get('top3', 0)),
                'overall_accuracy': float(metrics.get('overall_accuracy', 0)),
                'overall_precision': float(metrics.get('overall_precision', 0)),
                'overall_recall': float(metrics.get('overall_recall', 0)),
                'overall_f1': float(metrics.get('overall_f1', 0)),
                'macro_precision': float(metrics.get('macro_precision', 0)),
                'macro_recall': float(metrics.get('macro_recall', 0)),
                'macro_f1': float(metrics.get('macro_f1', 0)),
                'micro_precision': float(metrics.get('micro_precision', 0)),
                'micro_recall': float(metrics.get('micro_recall', 0)),
                'micro_f1': float(metrics.get('micro_f1', 0))
            },
            'grouped_accuracy': metrics.get('grouped', {})
        }
        
        # 添加到历史记录
        self.performance_history.append(task_result)
        
        # 保存到文件
        if hasattr(self.args, 'logfilename'):
            log_dir = os.path.dirname(self.args["logfilename"])
            
            # 保存为JSON
            json_path = os.path.join(log_dir, f"task_{task_id}_performance.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(task_result, f, indent=2, ensure_ascii=False)
            
            # 保存为CSV
            csv_data = []
            for i, perf in enumerate(self.performance_history):
                row = {
                    'task_id': perf['task_id'],
                    'total_classes': perf['total_classes'],
                    'known_classes': perf['known_classes']
                }
                row.update(perf['metrics'])
                csv_data.append(row)
            
            csv_df = pd.DataFrame(csv_data)
            csv_path = os.path.join(log_dir, "performance_history.csv")
            csv_df.to_csv(csv_path, index=False)
            
            logging.info(f"✅ 性能指标已保存: {json_path}")
            logging.info(f"✅ 性能历史已保存: {csv_path}")
    
    def evaluate_routing_network(self, data_loader, phase="test"):
        """
        独立的路由网络评估函数
        :param data_loader: 数据加载器（训练集或测试集）
        :param phase: 评估阶段，用于日志标识（"train" 或 "test"）
        :return: 路由评估结果字典
        """
        if self._cur_task < 1:  # 只有多任务时才需要评估路由
            logging.info(f"路由评估: 当前任务{self._cur_task}，需要至少2个任务")
            return None
        
        if not hasattr(self._network.convnet, 'moe_layer'):
            logging.info("路由评估: 未检测到MoE层")
            return None
        
        self._network.eval()
        
        # 初始化统计
        routing_stats = {
            'phase': phase,
            'total_samples': 0,
            'correct_routing': 0,
            'overall_accuracy': 0.0,
            'task_wise_accuracy': {},
            'expert_usage': {},
            'confusion_matrix': None
        }
        
        task_size = self.args["increment"]
        all_predicted_tasks = []
        all_true_tasks = []
        
        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(data_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                
                # 计算路由目标（与训练时相同的逻辑）
                routing_targets = self._compute_routing_targets(targets, task_size)
                
                # 前向传播
                outputs = self._network(inputs, task_id=None)
                gate_logits = outputs.get("gate_logits")
                
                if gate_logits is None:
                    continue
                
                # 预测的任务ID
                predicted_tasks = torch.argmax(gate_logits, dim=1)
                
                # 统计信息
                batch_correct = (predicted_tasks == routing_targets).sum().item()
                routing_stats['correct_routing'] += batch_correct
                routing_stats['total_samples'] += len(targets)
                
                # 保存详细数据用于后续分析
                all_predicted_tasks.extend(predicted_tasks.cpu().numpy())
                all_true_tasks.extend(routing_targets.cpu().numpy())
        
        # 计算总体指标
        if routing_stats['total_samples'] > 0:
            routing_stats['overall_accuracy'] = (
                routing_stats['correct_routing'] / routing_stats['total_samples']
            )
            
            # 计算任务级别的准确率
            all_predicted = np.array(all_predicted_tasks)
            all_true = np.array(all_true_tasks)
            
            for task_id in range(self._cur_task + 1):
                task_mask = all_true == task_id
                if task_mask.any():
                    task_acc = (all_predicted[task_mask] == all_true[task_mask]).mean()
                    routing_stats['task_wise_accuracy'][f'task_{task_id}'] = task_acc
            
            # 计算专家使用分布
            if len(all_predicted_tasks) > 0:
                unique, counts = np.unique(all_predicted_tasks, return_counts=True)
                total_predictions = len(all_predicted_tasks)
                for expert_id in range(self._cur_task + 1):
                    if expert_id in unique:
                        usage = counts[unique == expert_id][0] / total_predictions
                    else:
                        usage = 0.0
                    routing_stats['expert_usage'][f'expert_{expert_id}'] = usage
        
        # 打印评估结果
        self._print_routing_evaluation(routing_stats)
        
        # 保存评估历史（可选）
        self.routing_eval_history.append({
            'task': self._cur_task,
            'phase': phase,
            'stats': routing_stats
        })
        
        return routing_stats
    
    def _compute_routing_targets(self, targets, task_size):
        """
        计算路由目标（与训练时相同的逻辑）
        """
        routing_targets = torch.zeros_like(targets, device=self._device)
        
        # 新样本路由到当前专家
        new_sample_mask = (targets >= self._known_classes)
        routing_targets[new_sample_mask] = self._cur_task
        
        # 回放样本路由到对应专家
        replay_mask = (targets < self._known_classes)
        if replay_mask.any() and self._cur_task > 0:
            task_origin = targets // task_size
            task_origin = torch.clamp(task_origin, 0, self._cur_task - 1)
            routing_targets[replay_mask] = task_origin[replay_mask]
        
        return routing_targets
    
    def _print_routing_evaluation(self, routing_stats):
        """打印路由评估结果"""
        if routing_stats['total_samples'] == 0:
            return
        
        phase = routing_stats['phase']
        accuracy_pct = routing_stats['overall_accuracy'] * 100
        
        logging.info(f"\n🎯 {phase.upper()}集路由网络评估:")
        logging.info("-" * 50)
        logging.info(f"总体路由准确率: {accuracy_pct:.2f}% "
                    f"({routing_stats['correct_routing']}/{routing_stats['total_samples']})")
        
        # 任务级别准确率
        if routing_stats['task_wise_accuracy']:
            logging.info("\n任务级别路由准确率:")
            for task_name, acc in routing_stats['task_wise_accuracy'].items():
                logging.info(f"  {task_name}: {acc:.4f}")
        
        # 专家使用分布
        if routing_stats['expert_usage']:
            logging.info("\n专家使用分布:")
            for expert_name, usage in routing_stats['expert_usage'].items():
                logging.info(f"  {expert_name}: {usage:.4f}")
        
        logging.info("-" * 50)

def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]