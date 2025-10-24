import logging
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.inc_net import CosineIncrementalNet
from utils.toolkit import target2onehot, tensor2numpy

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


class iCaRL(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args["convnet_type"], False)

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

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

        if self.args['skip'] and self._cur_task==0:
            load_acc = self._network.load_checkpoint(self.args)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        if self._cur_task == 0:
            if self.args['skip']:
                self._network.to(self._device)
                cur_test_acc = self._compute_accuracy(self._network, self.test_loader)
                logging.info(f"Loaded_Test_Acc:{load_acc} Cur_Test_Acc:{cur_test_acc}")
            else:
                self._train(self.train_loader, self.test_loader) 
                self._compute_accuracy(self._network, self.test_loader)
        else:
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

                loss = loss_clf + loss_kd

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
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
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
def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
