import copy
import logging
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy, accuracy
from scipy.spatial.distance import cdist
import os
from sklearn.metrics import silhouette_score

EPSILON = 1e-8
batch_size = 64


class BaseLearner(object):
    def __init__(self, args):
        self.args = args
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self.topk = 1

        self._memory_size = args["memory_size"]
        self._memory_per_class = args.get("memory_per_class", None)
        self._fixed_memory = args.get("fixed_memory", False)
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]
        self.class_accuracy = {}  # 存储类别准确率

    @property
    def exemplar_size(self):
        assert len(self._data_memory) == len(
            self._targets_memory
        ), "Exemplar size error."
        return len(self._targets_memory)

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._total_classes

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim

    def build_rehearsal_memory(self, data_manager, per_class):
        if self._fixed_memory:
            self._construct_exemplar_unified(data_manager, per_class)
        else:
                    # 只有在有已知类别时才执行加权淘汰
            if self._known_classes > 0:
                self._reduce_exemplar_weight(data_manager, per_class)
                # self._reduce_exemplar(data_manager, per_class)
            else:
                self._reduce_exemplar(data_manager, per_class)
            # self._reduce_exemplar_weight(data_manager, per_class)
            self._construct_exemplar(data_manager, per_class)
            # self._construct_exemplar_new(data_manager, per_class)
        self._display_class_sample_counts()

    def save_checkpoint(self, test_acc):
        assert self.args['model_name'] == 'finetune'
        checkpoint_name = f"checkpoints/finetune_{self.args['csv_name']}"
        _checkpoint_cpu = copy.deepcopy(self._network)
        if isinstance(_checkpoint_cpu, nn.DataParallel):
            _checkpoint_cpu = _checkpoint_cpu.module
        _checkpoint_cpu.cpu()
        save_dict = {
            "tasks": self._cur_task,
            "convnet": _checkpoint_cpu.convnet.state_dict(),
            "fc":_checkpoint_cpu.fc.state_dict(),
            "test_acc": test_acc
        }
        torch.save(save_dict, "{}_{}.pkl".format(checkpoint_name, self._cur_task))
    
    def after_task(self):
        logging.info(f"update class accuracy")
        self._update_class_accuracy()
        logging.info(f"Class accuracy: {self.class_accuracy}")
    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes,self.args['increment'])
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )

        return ret

    def eval_task(self, save_conf=False):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None
        
        if save_conf:
            _pred = y_pred.T[0]
            _pred_path = os.path.join(self.args['logfilename'], "pred.npy")
            _target_path = os.path.join(self.args['logfilename'], "target.npy")
            np.save(_pred_path, _pred)
            np.save(_target_path, y_true)

            _save_dir = os.path.join(f"./results/conf_matrix/{self.args['prefix']}")
            os.makedirs(_save_dir, exist_ok=True)
            _save_path = os.path.join(_save_dir, f"{self.args['csv_name']}.csv")
            with open(_save_path, "a+") as f:
                f.write(f"{self.args['time_str']},{self.args['model_name']},{_pred_path},{_target_path} \n")
        
        return cnn_accy, nme_accy

    def incremental_train(self):
        pass

    def _train(self):
        pass

    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory)

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

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _eval_nme(self, loader, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        dists = cdist(class_means, vectors, "sqeuclidean")  # [nb_classes, N]
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance

        return np.argsort(scores, axis=1)[:, : self.topk], y_true  # [N, topk]

    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            if isinstance(self._network, nn.DataParallel):
                _vectors = tensor2numpy(
                    self._network.module.extract_vector(_inputs.to(self._device))
                )
            else:
                _vectors = tensor2numpy(
                    self._network.extract_vector(_inputs.to(self._device))
                )

            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def _reduce_exemplar(self, data_manager, m):
        logging.info("Reducing exemplars...({} per classes)".format(m))
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(
            self._targets_memory
        )
        self._class_means = np.zeros((self._total_classes, self.feature_dim))
        self._data_memory, self._targets_memory = np.array([]), np.array([])

        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            self._data_memory = (
                np.concatenate((self._data_memory, dd))
                if len(self._data_memory) != 0
                else dd
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, dt))
                if len(self._targets_memory) != 0
                else dt
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(dd, dt)
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar(self, data_manager, m):
        logging.info("Constructing exemplars...({} per classes)".format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection
                
                if len(vectors) == 0:
                    break
            # uniques = np.unique(selected_exemplars, axis=0)
            # print('Unique elements: {}'.format(len(uniques)))
            selected_exemplars = np.array(selected_exemplars)
            # exemplar_targets = np.full(m, class_idx)
            exemplar_targets = np.full(selected_exemplars.shape[0], class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar_unified(self, data_manager, m):
        logging.info(
            "Constructing exemplars for new classes...({} per classes)".format(m)
        )
        _class_means = np.zeros((self._total_classes, self.feature_dim))

        # Calculate the means of old classes with newly trained network
        for class_idx in range(self._known_classes):
            mask = np.where(self._targets_memory == class_idx)[0]
            class_data, class_targets = (
                self._data_memory[mask],
                self._targets_memory[mask],
            )

            class_dset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(class_data, class_targets)
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        # Construct exemplars for new classes and calculate the means
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, class_dset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )

            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            exemplar_dset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            exemplar_loader = DataLoader(
                exemplar_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(exemplar_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        self._class_means = _class_means
    def _construct_exemplar_new(self, data_manager, m):
        """改进的样本选择策略：结合核心特征、边界特征和多样性（使用GPU加速聚类）"""
        logging.info("Constructing exemplars with improved strategy...({} per classes)".format(m))
        start_time = time.time()
        total_classes = self._total_classes - self._known_classes
        logging.info(f"Total new classes to process: {total_classes}")
        
        for class_idx in range(self._known_classes, self._total_classes):
            class_start_time = time.time()
            logging.info(f"Processing class {class_idx}...")
            
            # 获取数据
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            logging.info(f"Class {class_idx}: Loaded {len(data)} samples.")
            
            # 如果样本数不足，全部选择
            if len(data) <= m:
                selected_exemplars = data
                exemplar_targets = np.full(len(data), class_idx)
                self._data_memory = (
                    np.concatenate((self._data_memory, selected_exemplars))
                    if len(self._data_memory) != 0
                    else selected_exemplars
                )
                self._targets_memory = (
                    np.concatenate((self._targets_memory, exemplar_targets))
                    if len(self._targets_memory) != 0
                    else exemplar_targets
                )
                logging.info(f"Class {class_idx}: All samples selected (less than or equal to m).")
                continue
            
            # 提取特征向量
            logging.info(f"Class {class_idx}: Extracting features...")
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            logging.info(f"Class {class_idx}: Features extracted. Shape: {vectors.shape}")
            
            # 归一化特征向量
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)
            logging.info(f"Class {class_idx}: Features normalized and class mean computed.")
            
            # 改进的样本选择策略
            selected_exemplars = []
            exemplar_vectors = []
            
            # 1. 选择距离类中心最近的样本（核心特征）
            logging.info(f"Class {class_idx}: Selecting core sample...")
            dist_to_center = np.linalg.norm(vectors - class_mean, axis=1)
            closest_idx = np.argmin(dist_to_center)
            selected_exemplars.append(np.array(data[closest_idx]))
            exemplar_vectors.append(np.array(vectors[closest_idx]))
            vectors = np.delete(vectors, closest_idx, axis=0)
            data = np.delete(data, closest_idx, axis=0)
            logging.info(f"Class {class_idx}: Core sample selected.")
            
            # 2. 选择距离类中心最远的样本（边界特征）
            if len(vectors) > 0:
                logging.info(f"Class {class_idx}: Selecting boundary sample...")
                dist_to_center = np.linalg.norm(vectors - class_mean, axis=1)
                farthest_idx = np.argmax(dist_to_center)
                selected_exemplars.append(np.array(data[farthest_idx]))
                exemplar_vectors.append(np.array(vectors[farthest_idx]))
                vectors = np.delete(vectors, farthest_idx, axis=0)
                data = np.delete(data, farthest_idx, axis=0)
                logging.info(f"Class {class_idx}: Boundary sample selected.")
            else:
                logging.info(f"Class {class_idx}: No vectors left for boundary sample.")
            
            # 3. 使用聚类选择剩余样本（多样性）
            if len(vectors) > 0 and m > len(selected_exemplars):
                logging.info(f"Class {class_idx}: Starting clustering for diversity samples...")
                remaining = m - len(selected_exemplars)
                logging.info(f"Class {class_idx}: Remaining samples to select: {remaining}")
                
                # 自动确定最佳聚类数量
                logging.info(f"Class {class_idx}: Auto-clustering...")
                cluster_start_time = time.time()
                best_k, best_centers, best_labels, silhouette_score_val = self._auto_cluster(vectors, max_clusters=remaining)
                cluster_time = time.time() - cluster_start_time
                logging.info(f"Class {class_idx}: Auto-clustering completed in {cluster_time:.2f} seconds.")
                
                if best_k is not None:
                    logging.info(f"Class {class_idx}: Auto-selected {best_k} clusters (silhouette={silhouette_score_val:.3f})")
                    
                    # 创建索引映射表
                    index_map = np.arange(len(vectors))
                    
                    # 创建一个列表来存储所有要删除的索引
                    indices_to_remove = []
                    
                    # 从每个簇中选择最接近中心的样本
                    for cluster_id in range(best_k):
                        # 获取当前簇的样本索引（基于原始聚类结果）
                        cluster_indices = np.where(best_labels == cluster_id)[0]
                        if len(cluster_indices) == 0:
                            logging.warning(f"Class {class_idx}: Cluster {cluster_id} has no samples.")
                            continue
                        
                        # 获取当前簇的样本在剩余向量中的实际索引
                        valid_indices = index_map[cluster_indices]
                        
                        # 获取簇内样本
                        cluster_vectors = vectors[valid_indices]
                        cluster_data = data[valid_indices]
                        
                        # 找到最接近簇中心的样本
                        cluster_center = best_centers[cluster_id]
                        dists = np.linalg.norm(cluster_vectors - cluster_center, axis=1)
                        best_idx_in_cluster = np.argmin(dists)
                        
                        # 获取实际索引
                        global_idx = valid_indices[best_idx_in_cluster]
                        
                        # 添加到选择
                        selected_exemplars.append(np.array(data[global_idx]))
                        exemplar_vectors.append(np.array(vectors[global_idx]))
                        
                        # 记录要删除的索引
                        indices_to_remove.append(global_idx)
                        
                        # 如果达到所需数量，提前退出
                        if len(selected_exemplars) >= m:
                            logging.info(f"Class {class_idx}: Reached m samples after cluster {cluster_id}.")
                            break
                    
                    # 一次性删除所有选中的样本
                    if indices_to_remove:
                        # 创建保留样本的掩码
                        mask = np.ones(len(vectors), dtype=bool)
                        mask[indices_to_remove] = False
                        
                        # 更新向量和数据数组
                        vectors = vectors[mask]
                        data = data[mask]
                else:
                    # 聚类效果不好，使用备用策略
                    logging.info(f"Class {class_idx}: Clustering not effective, using fallback strategy")
                    remaining = m - len(selected_exemplars)
                    for k in range(remaining):
                        if len(vectors) == 0:
                            logging.info(f"Class {class_idx}: No vectors left in fallback.")
                            break
                        
                        # 选择距离类中心最近的样本
                        dist_to_center = np.linalg.norm(vectors - class_mean, axis=1)
                        i = np.argmin(dist_to_center)
                        selected_exemplars.append(np.array(data[i]))
                        exemplar_vectors.append(np.array(vectors[i]))
                        vectors = np.delete(vectors, i, axis=0)
                        data = np.delete(data, i, axis=0)
                    logging.info(f"Class {class_idx}: Fallback strategy completed.")
            
            # 4. 如果仍未达到所需数量，使用原始Herding算法补充
            if len(selected_exemplars) < m and len(vectors) > 0:
                logging.info(f"Class {class_idx}: Using Herding algorithm for remaining samples...")
                remaining = m - len(selected_exemplars)
                for k in range(remaining):
                    if len(vectors) == 0:
                        break
                    
                    S = np.sum(exemplar_vectors, axis=0)
                    mu_p = (vectors + S) / (len(exemplar_vectors) + 1)
                    i = np.argmin(np.linalg.norm(class_mean - mu_p, axis=1))
                    selected_exemplars.append(np.array(data[i]))
                    exemplar_vectors.append(np.array(vectors[i]))
                    vectors = np.delete(vectors, i, axis=0)
                    data = np.delete(data, i, axis=0)
                logging.info(f"Class {class_idx}: Herding algorithm completed.")
            
            # 添加到样本库
            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(selected_exemplars.shape[0], class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )
            logging.info(f"Class {class_idx}: Selected {len(selected_exemplars)} samples. Total memory size: {len(self._targets_memory)}")
            
            # 计算类均值
            logging.info(f"Class {class_idx}: Computing class mean...")
            idx_dataset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean
            logging.info(f"Class {class_idx}: Class mean computed.")
            
            class_time = time.time() - class_start_time
            logging.info(f"Class {class_idx} processed in {class_time:.2f} seconds.")
        
        total_time = time.time() - start_time
        logging.info(f"Constructed exemplars for all classes in {total_time:.2f} seconds.")
    
    def _auto_cluster(self, vectors, max_clusters=10):
        """自动选择最佳聚类数量（使用归一化WCSS）"""
        logging.info("Auto clustering: starting with unified metric...")
        
        # 1. 参数检查
        n_samples = len(vectors)
        if n_samples <= 1:
            logging.info("Auto clustering: too few vectors (<=1), return None.")
            return None, None, None, None
        
        # 2. 计算k=1时的WCSS（基准值）
        center_k1 = np.mean(vectors, axis=0)
        wcss_k1 = np.sum(np.linalg.norm(vectors - center_k1, axis=1)**2)
        logging.info(f"WCSS for k=1: {wcss_k1:.4f}")
        
        # 3. 设置聚类数量范围
        min_clusters = 1
        max_clusters = 3
        
        if min_clusters >= max_clusters:
            logging.info(f"Auto clustering: min_clusters({min_clusters}) >= max_clusters({max_clusters}), return None.")
            return None, None, None, None
        
        best_score = -1
        best_k = min_clusters
        best_centers = None
        best_labels = None
        
        # 4. 尝试不同的聚类数量
        for k in range(min_clusters, max_clusters + 1):
            try:
                logging.info(f"Auto clustering: trying k={k}...")
                centers, labels = self._gpu_kmeans(vectors, k)
                
                # 计算WCSS
                wcss = 0
                for i in range(k):
                    cluster_mask = labels == i
                    if np.sum(cluster_mask) > 0:
                        cluster_points = vectors[cluster_mask]
                        wcss += np.sum(np.linalg.norm(cluster_points - centers[i], axis=1)**2)
                
                # 计算归一化WCSS分数
                normalized_score = 1 - (wcss / wcss_k1)
                logging.info(f"Auto clustering: k={k} WCSS: {wcss:.4f}, Normalized score: {normalized_score:.4f}")
                
                # 添加小聚类惩罚（可选）
                # 避免产生过小的簇
                min_cluster_size = np.min(np.bincount(labels))
                if min_cluster_size < 5:  # 如果最小簇小于5个样本
                    penalty = 0.1 * (5 - min_cluster_size)
                    normalized_score -= penalty
                    logging.info(f"Applied penalty: -{penalty:.4f} for small cluster")
                
                if normalized_score > best_score:
                    best_score = normalized_score
                    best_k = k
                    best_centers = centers
                    best_labels = labels
                    logging.info(f"Auto clustering: new best k={k} with score={normalized_score:.4f}")
            except Exception as e:
                logging.warning(f"Auto clustering failed for k={k}: {e}")
                continue
        
        # 5. 如果分数太低，认为聚类效果不好
        if best_score < 0.1:  # 阈值可根据实际情况调整
            logging.info(f"Auto clustering: best score {best_score:.4f} < 0.1, return None.")
            return None, None, None, None
        
        logging.info(f"Auto clustering: best k={best_k} with score={best_score:.4f}")
        return best_k, best_centers, best_labels, best_score
    
    def _gpu_kmeans(self, vectors, n_clusters, max_iter=100, tol=1e-4):
        """GPU加速的KMeans实现"""
        logging.info(f"GPU KMeans: starting with n_clusters={n_clusters}...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")
        
        vectors_tensor = torch.tensor(vectors, dtype=torch.float32).to(device)
        
        # 随机初始化聚类中心
        indices = torch.randperm(vectors_tensor.size(0))[:n_clusters]
        centers = vectors_tensor[indices]
        
        logging.info(f"GPU KMeans: initial centers initialized.")
        
        for iter in range(max_iter):
            # 计算每个样本到每个中心的距离
            dists = torch.cdist(vectors_tensor, centers)
            
            # 分配样本到最近的簇
            _, labels = torch.min(dists, dim=1)
            
            # 更新聚类中心
            new_centers = torch.zeros_like(centers)
            for i in range(n_clusters):
                mask = labels == i
                if mask.sum() > 0:
                    new_centers[i] = vectors_tensor[mask].mean(dim=0)
                else:
                    # 如果簇为空，随机选择一个样本作为中心
                    rand_idx = torch.randint(0, vectors_tensor.size(0), (1,))
                    new_centers[i] = vectors_tensor[rand_idx]
            
            # 检查收敛
            center_shift = torch.norm(centers - new_centers, dim=1).max()
            if center_shift <= tol:
                logging.info(f"GPU KMeans: converged at iteration {iter+1}.")
                break
                
            centers = new_centers
        
        if iter == max_iter - 1:
            logging.info(f"GPU KMeans: reached max iteration {max_iter}.")
        
        logging.info(f"GPU KMeans: completed in {iter+1} iterations.")
        return centers.cpu().numpy(), labels.cpu().numpy()
    
    def _reduce_exemplar_weight(self, data_manager, m):
        """基于准确率的加权淘汰机制（优化内存利用率）"""
        logging.info("Reducing exemplars with weighted strategy...")
        start_time = time.time()
        
        # 1. 备份当前样本库
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(
            self._targets_memory
        )
        
        # 2. 重置样本库和类均值
        self._class_means = np.zeros((self._total_classes, self.feature_dim))
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        
        # 3. 计算新类别将占用的内存
        new_classes_count = self._total_classes - self._known_classes
        new_classes_total = new_classes_count * m
        max_memory_for_old = self._memory_size - new_classes_total
        
        logging.info(f"Total memory: {self._memory_size}")
        logging.info(f"New classes: {new_classes_count}, will use {new_classes_total} samples")
        logging.info(f"Max memory for old classes: {max_memory_for_old}")
        
        # 4. 计算每个类别的准确率权重
        class_weights = self._calculate_accuracy_weights()
        total_weight = sum(class_weights.values())
        logging.info(f"Total weight: {total_weight:.4f}")
        
        # 5. 计算平均权重
        avg_weight = total_weight / len(class_weights) if len(class_weights) > 0 else 1.0
        
        # 6. 计算每个类别的最大可用样本数
        max_samples_per_class = {}
        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            max_samples_per_class[class_idx] = len(mask)
        
        # 7. 计算每个类别的加权样本数
        class_samples = {}
        total_allocated = 0
        
        # 第一轮：计算基本加权样本数
        for class_idx in range(self._known_classes):
            # 计算加权样本数
            weight_factor = class_weights[class_idx] / avg_weight
            weighted_m = int(m * weight_factor)
            
            # 确保至少保留1个样本，不超过最大可用样本数
            samples = max(1, min(weighted_m, max_samples_per_class[class_idx]))
            class_samples[class_idx] = samples
            total_allocated += samples
        
        # # 8. 调整总样本数以充分利用内存
        # if total_allocated < max_memory_for_old:
        #     # 计算可用剩余空间
        #     remaining_space = max_memory_for_old - total_allocated
        #     logging.info(f"Remaining memory space for old classes: {remaining_space}, allocating extra samples...")
            
        #     # 计算可分配权重的总和（只考虑还有额外样本的类别）
        #     total_remaining_weight = 0
        #     for class_idx in class_samples:
        #         if class_samples[class_idx] < max_samples_per_class[class_idx]:
        #             total_remaining_weight += class_weights[class_idx]
            
        #     # 如果还有权重可分配
        # if total_allocated < max_memory_for_old:
        #     # 计算可用剩余空间
        #     remaining_space = max_memory_for_old - total_allocated
        #     logging.info(f"Remaining memory space for old classes: {remaining_space}, allocating extra samples...")
            
        #     # 计算可分配权重的总和（只考虑还有额外样本的类别）
        #     total_remaining_weight = 0
        #     for class_idx in class_samples:
        #         if class_samples[class_idx] < max_samples_per_class[class_idx]:
        #             total_remaining_weight += class_weights[class_idx]
            
        #     # 如果还有权重可分配
        #     if total_remaining_weight > 0:
        #         # 按比例分配剩余空间
        #         for class_idx in class_samples:
        #             # 检查是否还有额外样本可用
        #             if class_samples[class_idx] < max_samples_per_class[class_idx]:
        #                 # 计算额外样本数
        #                 extra_samples = int(remaining_space * class_weights[class_idx] / total_remaining_weight)
                        
        #                 # 确保不超过最大可用样本数
        #                 max_possible = max_samples_per_class[class_idx] - class_samples[class_idx]
        #                 actual_extra = min(extra_samples, max_possible)
                        
        #                 # 更新样本数
        #                 class_samples[class_idx] += actual_extra
        #                 total_allocated += actual_extra
        #                 remaining_space -= actual_extra
                        
        #                 if actual_extra > 0:
        #                     logging.info(f"Class {class_idx}: Added {actual_extra} extra samples")
                
            # 更新剩余空间
            # if remaining_space > 0:
            #     logging.info(f"After allocation, {remaining_space} samples still remain unused.")
        
        # 9. 检查并调整总样本数不超过内存限制
        if total_allocated > max_memory_for_old:
            logging.info(f"Total allocated {total_allocated} exceeds max memory for old classes {max_memory_for_old}, scaling down...")
            scale = max_memory_for_old / total_allocated
            for class_idx in class_samples:
                new_samples = max(1, int(class_samples[class_idx] * scale))
                class_samples[class_idx] = new_samples
        
        # 10. 保留样本
        total_final = 0
        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            samples = class_samples[class_idx]
            dd, dt = dummy_data[mask][:samples], dummy_targets[mask][:samples]
            
            # 添加到样本库
            self._data_memory = np.concatenate((self._data_memory, dd)) if len(self._data_memory) != 0 else dd
            self._targets_memory = np.concatenate((self._targets_memory, dt)) if len(self._targets_memory) != 0 else dt
            total_final += samples
            
            logging.info(f"Class {class_idx}: Retained {samples} samples (weight={class_weights[class_idx]:.4f})")
            
            # 计算类均值
            self._compute_class_mean(data_manager, dd, dt, class_idx)
        
        # 11. 打印权重详情
        self._print_weight_details(class_weights, class_samples)
        
        total_time = time.time() - start_time
        logging.info(f"Accuracy-based reduction completed in {total_time:.2f} seconds")
        logging.info(f"Total samples for old classes: {total_final}/{max_memory_for_old} ({(total_final/max_memory_for_old)*100:.2f}%)")
        
        # 12. 计算总内存使用预测
        total_samples_predicted = total_final + new_classes_total
        memory_percentage = (total_samples_predicted / self._memory_size) * 100
        logging.info(f"Predicted total memory usage: {total_samples_predicted}/{self._memory_size} ({memory_percentage:.2f}%)")
    def _calculate_accuracy_weights(self):
        """基于准确率的平滑权重计算"""
        logging.info("Calculating smoothed accuracy weights...")
        class_weights = {}
        
        # 1. 获取每个类别的准确率
        accuracies = []
        for class_idx in range(self._known_classes):
            if hasattr(self, 'class_accuracy') and class_idx in self.class_accuracy:
                accuracy = self.class_accuracy[class_idx]
            else:
                accuracy = 0.5  # 默认值
            accuracies.append(accuracy)
        
        # 2. 计算平均准确率
        avg_accuracy = np.mean(accuracies) if accuracies else 0.5
        factor = self.args['replay_factor']
        # 3. 计算权重
        for class_idx, accuracy in enumerate(accuracies):
            # 计算与平均准确率的偏差
            deviation = accuracy - avg_accuracy
            
            # 使用平滑函数计算权重
            # 当准确率接近平均值时，权重接近1.0
            # 当准确率偏离平均值时，权重在0.9-1.1之间变化
            weight = 1.0 - factor * np.tanh(deviation * 2)
            
            class_weights[class_idx] = weight
            logging.info(f"Class {class_idx}: Accuracy={accuracy:.4f}, Weight={weight:.4f}")
        
        return class_weights

    def _print_weight_details(self, class_weights, class_samples):
        """打印权重详情"""
        logging.info("Accuracy-based reduction details:")
        logging.info("Class | Accuracy | Weight | Weight Factor | Allocated Samples")
        for class_idx in class_weights:
            accuracy = self.class_accuracy.get(class_idx, 0.5)
            weight = class_weights[class_idx]
            samples = class_samples[class_idx]
            
            # 计算权重因子
            avg_weight = sum(class_weights.values()) / len(class_weights)
            weight_factor = weight / avg_weight
            
            logging.info(f"{class_idx:5} | {accuracy:.4f} | {weight:.4f} | {weight_factor:.4f} | {samples:15}")
    
    def _compute_class_mean(self, data_manager, data, targets, class_idx):
        """计算类均值"""
        idx_dataset = data_manager.get_dataset(
            [], source="train", mode="test", appendent=(data, targets)
        )
        idx_loader = DataLoader(
            idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )
        vectors, _ = self._extract_vectors(idx_loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
        mean = np.mean(vectors, axis=0)
        mean = mean / np.linalg.norm(mean)
        
        self._class_means[class_idx, :] = mean
    
    def _update_class_accuracy(self):
        """在测试集上计算并更新类别准确率"""
        # 确保有测试集加载器
        if not hasattr(self, 'test_loader') or self.test_loader is None:
            logging.warning("Test loader not available, skip updating class accuracy.")
            return
        
        # 在测试集上评估
        y_pred, y_true = self._eval_cnn(self.test_loader)
        
        # 计算每个类别的准确率
        class_correct = {}
        class_total = {}
        
        for i in range(len(y_true)):
            class_idx = y_true[i]
            class_total[class_idx] = class_total.get(class_idx, 0) + 1
            if y_pred[i, 0] == class_idx:  # 检查top1预测
                class_correct[class_idx] = class_correct.get(class_idx, 0) + 1
        
        # 更新准确率
        new_accuracy = {}
        for class_idx in class_total:
            accuracy = class_correct.get(class_idx, 0) / class_total[class_idx]
            new_accuracy[class_idx] = accuracy
            logging.info(f"Class {class_idx}: Test accuracy updated to {accuracy:.4f}")
        
        # 合并新旧准确率
        self.class_accuracy.update(new_accuracy)
    def _display_class_sample_counts(self):
        """显示每类样本数量"""
        logging.info("Displaying class sample counts after building rehearsal memory...")
        
        # 如果没有样本，直接返回
        if len(self._targets_memory) == 0:
            logging.info("Memory is empty.")
            return
        
        # 统计每个类别的样本数量
        class_counts = {}
        for class_idx in np.unique(self._targets_memory):
            mask = self._targets_memory == class_idx
            count = np.sum(mask)
            class_counts[class_idx] = count
        
        # 打印结果
        logging.info("Class | Sample Count")
        for class_idx in sorted(class_counts.keys()):
            count = class_counts[class_idx]
            logging.info(f"{class_idx:5} | {count:12}")
        
        # 打印总数
        total_samples = len(self._targets_memory)
        logging.info(f"Total samples: {total_samples}")
        
        # 打印内存使用情况
        memory_percentage = (total_samples / self._memory_size) * 100
        logging.info(f"Memory usage: {total_samples}/{self._memory_size} ({memory_percentage:.2f}%)")