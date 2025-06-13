import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

import os
from torchvision import datasets, transforms
import numpy as np

class USTC2016:
    """USTC2016网络流量图像数据集类（使用ImageFolder实现）"""
    use_path = True  # 使用路径而不是加载整个图像到内存
    
    # 数据增强配置 - 保持与CIFAR10类似
    train_trsf = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
    ]
    test_trsf = []
    common_trsf = [
        transforms.Resize((32, 32)),  # 调整为32×32
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
    
    class_order = None  # 将在download_data中初始化
    
    def __init__(self):
        self.root = "./data"  # 数据集根目录
        self.train_data = None
        self.train_targets = None
        self.test_data = None
        self.test_targets = None
        
    def download_data(self):
        """加载USTC2016数据集 - 使用ImageFolder实现"""
        # 构建数据集路径
        train_dir = os.path.join(self.root, "ustc2016/final_data/train")
        test_dir = os.path.join(self.root, "ustc2016/final_data/test")
        
        # 使用ImageFolder加载训练数据
        train_dataset = datasets.ImageFolder(
            train_dir,
            transform=transforms.ToTensor()  # 仅用于获取图像和标签
        )
        
        # 使用ImageFolder加载测试数据
        test_dataset = datasets.ImageFolder(
            test_dir,
            transform=transforms.ToTensor()  # 仅用于获取图像和标签
        )
        
        # 获取类顺序（按文件夹名称排序）
        self.class_order = train_dataset.classes
        
        # 提取图像路径和标签
        self.train_data = np.array([s[0] for s in train_dataset.samples])  # 文件路径
        self.train_targets = np.array([s[1] for s in train_dataset.samples])  # 标签索引
        
        self.test_data = np.array([s[0] for s in test_dataset.samples])  # 文件路径
        self.test_targets = np.array([s[1] for s in test_dataset.samples])  # 标签索引
        
        print(f"已加载 USTC2016 数据集:")
        print(f"  训练集大小: {len(self.train_data)}")
        print(f"  测试集大小: {len(self.test_data)}")
        print(f"  类别数量: {len(self.class_order)}")