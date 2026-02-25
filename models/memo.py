import logging
import os
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import copy
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import AdaptiveNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score

num_workers=8
T=2

class MEMO(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._old_base = None
        self._network = AdaptiveNet(args['convnet_type'], False)
        logging.info(f'>>> train generalized blocks:{self.args["train_base"]} train_adaptive:{self.args["train_adaptive"]}')

    def after_task(self):
        self._known_classes = self._total_classes
        if self._cur_task == 0:
            if self.args['train_base']:
                logging.info("Train Generalized Blocks...")
                self._network.TaskAgnosticExtractor.train()
                for param in self._network.TaskAgnosticExtractor.parameters():
                    param.requires_grad = True
            else:
                logging.info("Fix Generalized Blocks...")
                self._network.TaskAgnosticExtractor.eval()
                for param in self._network.TaskAgnosticExtractor.parameters():
                    param.requires_grad = False
        super().after_task()
        self._compute_detailed_metrics_for_loader(self.test_loader)
        logging.info('Exemplar size: {}'.format(self.exemplar_size))
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
        return 
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
                                f"confusion_memo_task_{self._cur_task}.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()
        return
    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)

        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        if self._cur_task>0:
            for i in range(self._cur_task):
                for p in self._network.AdaptiveExtractors[i].parameters():
                    if self.args['train_adaptive']:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False

        logging.info('All params: {}'.format(count_parameters(self._network)))
        logging.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source='train',
            mode='train', 
            appendent=self._get_memory()
        )
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.args["batch_size"], 
            shuffle=True, 
            num_workers=num_workers
        )
        
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), 
            source='test', 
            mode='test'
        )
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=self.args["batch_size"],
            shuffle=False, 
            num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
    
    def set_network(self):
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        self._network.train()                   #All status from eval to train
        if self.args['train_base']:
            self._network.TaskAgnosticExtractor.train()
        else:
            self._network.TaskAgnosticExtractor.eval()
        
        # set adaptive extractor's status
        self._network.AdaptiveExtractors[-1].train()
        if self._cur_task >= 1:
            for i in range(self._cur_task):
                if self.args['train_adaptive']:
                    self._network.AdaptiveExtractors[i].train()
                else:
                    self._network.AdaptiveExtractors[i].eval()
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
            
    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task==0:
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                momentum=0.9,
                lr=self.args["init_lr"],
                weight_decay=self.args["init_weight_decay"]
            )
            if self.args['scheduler'] == 'steplr':
                scheduler = optim.lr_scheduler.MultiStepLR(
                    optimizer=optimizer, 
                    milestones=self.args['init_milestones'], 
                    gamma=self.args['init_lr_decay']
                )
            elif self.args['scheduler'] == 'cosine':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer,
                    T_max=self.args['init_epoch']
                ) 
            else:
                raise NotImplementedError
            
            if not self.args['skip']:
                self._init_train(train_loader, test_loader, optimizer, scheduler)
            else:
                if isinstance(self._network, nn.DataParallel):
                    self._network = self._network.module
                load_acc = self._network.load_checkpoint(self.args)
                self._network.to(self._device)

                if len(self._multiple_gpus) > 1:
                    self._network = nn.DataParallel(self._network, self._multiple_gpus)
                
                cur_test_acc = self._compute_accuracy(self._network, self.test_loader)
                logging.info(f"Loaded_Test_Acc:{load_acc} Cur_Test_Acc:{cur_test_acc}")
        else:
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()), 
                lr=self.args['lrate'], 
                momentum=0.9, 
                weight_decay=self.args['weight_decay']
            )
            if self.args['scheduler'] == 'steplr':
                scheduler = optim.lr_scheduler.MultiStepLR(
                    optimizer=optimizer,
                    milestones=self.args['milestones'], 
                    gamma=self.args['lrate_decay']
                )
            elif self.args['scheduler'] == 'cosine':
                assert self.args['t_max'] is not None
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer,
                    T_max=self.args['t_max']
                )
            else:
                raise NotImplementedError
            self._update_representation(train_loader, test_loader, optimizer, scheduler)
            if len(self._multiple_gpus) > 1:
                self._network.module.weight_align(self._total_classes-self._known_classes)
            else:
                self._network.weight_align(self._total_classes-self._known_classes)

            
    def _init_train(self,train_loader,test_loader,optimizer,scheduler):
        prog_bar = tqdm(range(self.args["init_epoch"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)['logits']

                loss=F.cross_entropy(logits,targets) 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            if epoch%5==0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch+1, self.args['init_epoch'], losses/len(train_loader), train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, epoch+1, self.args['init_epoch'], losses/len(train_loader), train_acc)
            # prog_bar.set_description(info)
            logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["epochs"]))
        for _, epoch in enumerate(prog_bar):
            self.set_network()
            losses = 0.
            losses_clf=0.
            losses_aux=0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                outputs= self._network(inputs)
                logits,aux_logits=outputs["logits"],outputs["aux_logits"]
                loss_clf=F.cross_entropy(logits,targets)
                aux_targets = targets.clone()
                aux_targets=torch.where(aux_targets-self._known_classes+1>0,  aux_targets-self._known_classes+1,0)
                loss_aux=F.cross_entropy(aux_logits,aux_targets)
                loss=loss_clf+self.args['alpha_aux']*loss_aux

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_aux+=loss_aux.item()
                losses_clf+=loss_clf.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            if epoch%5==0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux  {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch+1, self.args["epochs"], losses/len(train_loader),losses_clf/len(train_loader),losses_aux/len(train_loader),train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, epoch+1, self.args["epochs"], losses/len(train_loader), losses_clf/len(train_loader),losses_aux/len(train_loader),train_acc)
            prog_bar.set_description(info)
        logging.info(info)