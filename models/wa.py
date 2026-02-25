import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import target2onehot, tensor2numpy
import pandas as pd
import json
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt

EPSILON = 1e-8


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


class WA(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = IncrementalNet(args["convnet_type"], False, use_moe=True)
        self.performance_history = []  # 存储每个任务的性能指标历史

    def after_task(self):
        """在任务学习后执行，包括权重对齐和性能评估"""
        if self._cur_task > 0:
            self._network.weight_align(self._total_classes - self._known_classes)
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        
        # 在每个任务后进行评估
        self._evaluate_after_task()
        
        logging.info("Exemplar size: {}".format(self.exemplar_size))
        super().after_task()
    
    def _evaluate_after_task(self):
        """在after_task中评估模型性能"""
        logging.info(f"\n{'='*60}")
        logging.info(f"任务 {self._cur_task} 完成后的性能评估")
        logging.info(f"{'='*60}")
        
        # 评估当前模型在所有已见类别上的性能
        if hasattr(self, 'test_loader'):
            detailed_metrics = self._compute_detailed_metrics_for_loader(self.test_loader)
            
            # 保存性能指标
            self._save_task_performance(detailed_metrics, self._cur_task)
            
            # 打印详细指标
            self._print_detailed_metrics(detailed_metrics)
        
        # 如果有旧网络，也可以对比评估
        if self._old_network is not None and self._cur_task > 0:
            self._evaluate_forgetting()
    
    def _compute_detailed_metrics_for_loader(self, loader):
        """为数据加载器计算详细性能指标"""
        self._network.eval()
        
        all_preds = []
        all_targets = []
        all_logits = []
        
        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(loader):
                inputs = inputs.to(self._device)
                outputs = self._network(inputs)
                logits = outputs["logits"]
                preds = torch.max(logits, dim=1)[1]
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_logits.extend(logits.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_logits = np.vstack(all_logits)
        self._compute_confustion_then_save(all_preds, all_targets)
        return self._compute_detailed_metrics(all_preds, all_targets, all_logits)
    def _compute_confustion_then_save(self, y_pred, y_true):
        from utils.data_manager import shuffled_class_order
        class_labels = shuffled_class_order[:self._total_classes]
        confusion = confusion_matrix(y_true, y_pred)
        
        # 保存原始数据
        np.save(os.path.join(self.args["logfilename"], 
                            f"confusion_wa_moe_task_{self._cur_task}.npy"), 
                confusion)
                    # === 加入这几行 ===
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
        plt.rcParams['axes.unicode_minus'] = False
        # 保存可视化图片
        plt.figure(figsize=(10, 8))
        plt.imshow(confusion, cmap='Blues', interpolation='nearest')
        tick_marks = np.arange(len(class_labels))
        plt.xticks(tick_marks, class_labels, rotation=45, fontsize=12)
        plt.yticks(tick_marks, class_labels, rotation=45,fontsize=12)
        # 添加数值标注
        for i in range(confusion.shape[0]):
            for j in range(confusion.shape[1]):
                plt.text(j, i, str(confusion[i, j]),
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=12)
        
        plt.colorbar()
        plt.xlabel('预测类别', fontsize=14)
        plt.ylabel('真实类别', fontsize=14)
        # plt.title(f'任务 {self._cur_task} 混淆矩阵', fontsize=16)

        # 调整布局并保存
        plt.tight_layout()
        plt.savefig(os.path.join(self.args["logfilename"],
                                f"confusion_wa_task_{self._cur_task}.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()
        return
    def _compute_detailed_metrics(self, y_pred, y_true, y_logits):
        """计算详细的性能指标"""
        metrics = {}
        
        # 基础指标
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # 类别数量
        n_classes = len(np.unique(y_true))
        
        # 计算加权平均指标
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # 计算宏平均指标
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # 计算微平均指标
        metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        
        # 生成分类报告
        class_report = classification_report(
            y_true, y_pred, 
            output_dict=True,
            zero_division=0
        )
        metrics['classification_report'] = class_report
        
        # 计算每个类别的指标
        per_class_metrics = {}
        for class_idx in range(n_classes):
            class_mask = y_true == class_idx
            if np.sum(class_mask) > 0:  # 只计算有样本的类别
                per_class_metrics[class_idx] = {
                    'precision': precision_score(y_true, y_pred, average=None, zero_division=0)[class_idx],
                    'recall': recall_score(y_true, y_pred, average=None, zero_division=0)[class_idx],
                    'f1': f1_score(y_true, y_pred, average=None, zero_division=0)[class_idx],
                    'support': int(np.sum(class_mask))
                }
        
        metrics['per_class_metrics'] = per_class_metrics
        
        # 计算top-1和top-3准确率
        metrics['top1'] = metrics['accuracy']
        
        # 计算top-3准确率
        top3_correct = 0
        for i in range(len(y_true)):
            top3 = np.argsort(y_logits[i])[-3:][::-1]  # 降序取前3
            if y_true[i] in top3:
                top3_correct += 1
        metrics['top3'] = top3_correct / len(y_true)
        
        # 计算混淆矩阵
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        
        return metrics
    
    def _print_detailed_metrics(self, metrics):
        """打印详细的性能指标"""
        logging.info(f"📊 详细性能指标 (任务 {self._cur_task}):")
        logging.info("-" * 60)
        logging.info(f"{'指标':<25} | {'值':<10}")
        logging.info("-" * 60)
        logging.info(f"{'Top-1准确率':<25} | {metrics['accuracy']:.4f}")
        logging.info(f"{'Top-3准确率':<25} | {metrics['top3']:.4f}")
        logging.info("-" * 60)
        logging.info(f"{'加权精确率':<25} | {metrics['precision_weighted']:.4f}")
        logging.info(f"{'加权召回率':<25} | {metrics['recall_weighted']:.4f}")
        logging.info(f"{'加权F1分数':<25} | {metrics['f1_weighted']:.4f}")
        logging.info("-" * 60)
        logging.info(f"{'宏平均精确率':<25} | {metrics['precision_macro']:.4f}")
        logging.info(f"{'宏平均召回率':<25} | {metrics['recall_macro']:.4f}")
        logging.info(f"{'宏平均F1分数':<25} | {metrics['f1_macro']:.4f}")
        logging.info("-" * 60)
        logging.info(f"{'微平均精确率':<25} | {metrics['precision_micro']:.4f}")
        logging.info(f"{'微平均召回率':<25} | {metrics['recall_micro']:.4f}")
        logging.info(f"{'微平均F1分数':<25} | {metrics['f1_micro']:.4f}")
        logging.info("=" * 60)
        
        # 打印类别数量信息
        n_classes = len(metrics['per_class_metrics'])
        logging.info(f"评估类别数: {n_classes}")
        # logging.info(f"总样本数: {len(metrics.get('classification_report', 0).get('accuracy', {}).get('support', 0))}")
        
        # 打印分类报告摘要
        if 'classification_report' in metrics:
            report = metrics['classification_report']
            logging.info("\n📋 分类报告摘要 (前5个类别):")
            logging.info("-" * 60)
            
            count = 0
            for key, value in report.items():
                if key not in ['accuracy', 'macro avg', 'weighted avg'] and isinstance(value, dict):
                    logging.info(f"类别 {key}: 精确率={value['precision']:.4f}, 召回率={value['recall']:.4f}, F1={value['f1-score']:.4f}, 支持数={value['support']}")
                    count += 1
                    if count >= 5:  # 只显示前5个类别
                        break
    
    def _save_task_performance(self, metrics, task_id):
        """保存当前任务的性能指标"""
        # 创建结果字典
        task_result = {
            'task_id': task_id,
            'total_classes': self._total_classes,
            'known_classes': self._known_classes,
            'timestamp': pd.Timestamp.now().isoformat(),
            'metrics': {
                'top1_accuracy': float(metrics.get('accuracy', 0)),
                'top3_accuracy': float(metrics.get('top3', 0)),
                'precision_weighted': float(metrics.get('precision_weighted', 0)),
                'recall_weighted': float(metrics.get('recall_weighted', 0)),
                'f1_weighted': float(metrics.get('f1_weighted', 0)),
                'precision_macro': float(metrics.get('precision_macro', 0)),
                'recall_macro': float(metrics.get('recall_macro', 0)),
                'f1_macro': float(metrics.get('f1_macro', 0)),
                'precision_micro': float(metrics.get('precision_micro', 0)),
                'recall_micro': float(metrics.get('recall_micro', 0)),
                'f1_micro': float(metrics.get('f1_micro', 0))
            }
        }
        
        # 添加到历史记录
        self.performance_history.append(task_result)
        
        # 保存到文件
        if hasattr(self.args, 'logfilename'):
            log_dir = os.path.dirname(self.args["logfilename"])
            
            # 保存为JSON
            json_path = os.path.join(log_dir, f"wa_task_{task_id}_performance.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(task_result, f, indent=2, ensure_ascii=False)
            
            # 保存为CSV
            if self.performance_history:
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
                csv_path = os.path.join(log_dir, "wa_performance_history.csv")
                csv_df.to_csv(csv_path, index=False)
                
                logging.info(f"✅ WA性能指标已保存: {json_path}")
                logging.info(f"✅ WA性能历史已保存: {csv_path}")
    
    def _evaluate_forgetting(self):
        """评估遗忘情况（旧任务性能）"""
        if self._cur_task < 1 or self._old_network is None:
            return
        
        logging.info("\n🔍 遗忘评估 (旧任务性能):")
        logging.info("-" * 50)
        
        # 这里可以添加对旧任务性能的评估逻辑
        # 例如，比较当前模型和旧网络在旧任务上的性能差异
        
        # 获取旧任务的类别范围
        old_classes = np.arange(0, self._known_classes - self.args.get("increment", 10))
        if len(old_classes) == 0:
            return
        
        # 评估当前模型在旧任务上的性能
        self._network.eval()
        self._old_network.eval()
        
        # 这里可以添加具体的遗忘评估逻辑
        # 例如，比较两个模型在旧类别上的准确率差异
        
        logging.info(f"评估旧任务类别: {old_classes[0]} 到 {old_classes[-1]}")
        logging.info("-" * 50)
        
    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        # Loader
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        # Procedure
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

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
            )  # 1e-5
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=milestones, gamma=lrate_decay
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)
            if len(self._multiple_gpus) > 1:
                self._network.module.weight_align(
                    self._total_classes - self._known_classes
                )
            else:
                self._network.weight_align(self._total_classes - self._known_classes)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(init_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

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
                # 使用详细评估函数
                test_metrics = self._compute_detailed_accuracy(self._network, test_loader)
                test_acc = test_metrics['accuracy']
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
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss_clf = F.cross_entropy(logits, targets)
                loss_kd = _KD_loss(
                    logits[:, : self._known_classes],
                    self._old_network(inputs)["logits"],
                    T,
                )

                loss = 3*loss_clf +  loss_kd

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                # acc
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                # 使用详细评估函数
                test_metrics = self._compute_detailed_accuracy(self._network, test_loader)
                test_acc = test_metrics['accuracy']
                test_f1 = test_metrics['f1_score']
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}, Test_F1 {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                    test_f1,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)
    
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
                outputs = model(inputs)
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
                outputs = model(inputs)
                logits = outputs["logits"]
            predicts = torch.max(logits, dim=1)[1]
            all_preds.extend(predicts.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # 计算各种指标
        accuracy = accuracy_score(all_targets, all_preds) * 100
        precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0) * 100
        recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0) * 100
        f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0) * 100
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]