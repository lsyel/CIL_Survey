# icarl.py （完整修改版）

import logging
import numpy as np
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

EPSILON = 1e-8

# ========== 超参数 ==========
init_epoch = 50
init_lr = 0.1
init_milestones = [60, 120, 170]
init_lr_decay = 0.1
init_weight_decay = 0.0005

epochs = 30
lrate = 0.1
milestones = [80, 120]
lrate_decay = 0.1
batch_size = 128
weight_decay = 2e-4
num_workers = 8
T = 2  # 蒸馏温度


class iCaRL(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        # 👇 初始化网络时传入 use_moe 参数
        self._network = IncrementalNet(
            args["convnet_type"],
            False,
            gradcam=False,
            use_moe=args.get("use_moe", True)  # 关键：启用 MoE
        )
        self._cur_task = -1  # 初始化为 -1，第一个任务变成 0

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
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                # 👇 课程学习：前80%强制路由，后20%自动路由
                if epoch < int(epochs * 0.8):
                    task_id_for_train = self._cur_task
                else:
                    task_id_for_train = None  # 自动路由
                # 👇 传入当前 task_id，控制 MoE 路由到当前任务专家
                output = self._network(inputs, task_id=task_id_for_train)
                logits = output["logits"]

                # 分类损失
                loss_clf = F.cross_entropy(logits, targets)

                # 蒸馏损失（旧类别部分）
                if self._old_network is not None:
                    with torch.no_grad():
                        old_output = self._old_network(inputs, None)  # 👈 也传 task_id
                    loss_kd = _KD_loss(
                        logits[:, : self._known_classes],
                        old_output["logits"],
                        T,
                    )
                else:
                    loss_kd = 0

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
        cnn_pred_list, cnn_target_list, cnn_logits_list = [], [], []  # 👈 新增 logits 列表
        self._network.eval()

        for i, (_, inputs, targets) in enumerate(self.test_loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs, task_id=None)
                logits = outputs["logits"]  # 👈 获取 logits
                cnn_logits_list.append(logits.cpu().numpy())  # 👈 保存 logits
            cnn_preds = torch.max(logits, dim=1)[1]

            cnn_pred_list.append(cnn_preds.cpu().numpy())
            cnn_target_list.append(targets.cpu().numpy())

        cnn_pred_all = np.concatenate(cnn_pred_list)
        cnn_target_all = np.concatenate(cnn_target_list)
        cnn_logits_all = np.vstack(cnn_logits_list)  # 👈 合并 logits

        # 👇 传入 4 个参数：pred, true, logits, known_classes
        cnn_accy = self._evaluate(cnn_pred_all, cnn_target_all, cnn_logits_all, self._known_classes)

        nme_accy = None

        if save_conf:
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
    def _evaluate(self, y_pred, y_true, y_logits, known_classes):
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
        for i in range(0, known_classes, task_size):
            mask = (y_true >= i) & (y_true < i + task_size)
            if mask.any():
                grouped[f"{i}-{i+task_size}"] = (y_pred[mask] == y_true[mask]).mean()
        ret["grouped"] = grouped

        return ret


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