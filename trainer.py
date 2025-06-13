import copy
import datetime
import json
import logging
import os
import sys
import time

import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import ConfigEncoder, count_parameters, save_fc, save_model

def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)

def _train(args):
    """
    执行模型的训练流程，包括实验配置初始化、模型训练、评估和结果保存等操作。

    :param args: 包含训练配置信息的字典
    """
    # 获取当前时间并格式化为特定字符串，去掉最后 3 位微秒数，方便标识实验
    time_str = datetime.datetime.now().strftime('%m%d-%H-%M-%S-%f')[:-3]
    args['time_str'] = time_str
    
    # 若初始类别数等于增量类别数，将初始类别数设为 0，否则使用配置的初始类别数
    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    # 根据实验参数生成实验名称
    exp_name = "{}_{}_{}_{}_B{}_Inc{}".format(
        args["time_str"],
        args["dataset"],
        args["convnet_type"],
        args["seed"],
        init_cls,
        args["increment"],
    )
    args['exp_name'] = exp_name

    # 根据 debug 模式确定日志文件的路径
    if args['debug']:
        logfilename = "logs/debug/{}/{}/{}/{}".format( 
            args["prefix"],
            args["dataset"],
            args["model_name"],
            args["exp_name"]
        )
    else:
        logfilename = "logs/{}/{}/{}/{}".format( 
            args["prefix"],
            args["dataset"],
            args["model_name"],
            args["exp_name"]
        )

    args['logfilename'] = logfilename

    # 生成用于保存结果的 CSV 文件名称
    csv_name = "{}_{}_{}_B{}_Inc{}".format( 
        args["dataset"],
        args["seed"],
        args["convnet_type"],
        init_cls,
        args["increment"],
    )
    args['csv_name'] = csv_name
    # 创建日志文件目录，若目录已存在则不报错
    os.makedirs(logfilename, exist_ok=True)

    # 拼接日志文件的完整路径
    log_path = os.path.join(args["logfilename"], "main.log")
    # 配置日志记录，同时输出到文件和标准输出，设置日志级别和格式
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # 记录当前时间字符串
    logging.info(f"Time Str >>> {args['time_str']}")
    # 保存配置信息到 JSON 文件
    config_filepath = os.path.join(args["logfilename"], 'configs.json')
    with open(config_filepath, "w") as fd:
        json.dump(args, fd, indent=2, sort_keys=True, cls=ConfigEncoder)

    # 设置随机种子，保证实验可重复性
    _set_random()
    # 设置训练设备
    _set_device(args)
    # 打印所有实验参数
    print_args(args)

    # 初始化数据管理器，用于管理数据集
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
    )
    # 通过工厂函数获取模型实例
    model = factory.get_model(args["model_name"], args)

    # 初始化 CNN 和 NME 的准确率曲线，以及是否有 NME 指标的标志
    cnn_curve, nme_curve, no_nme = {"top1": [], "top3": []}, {"top1": [], "top3": []}, True
    # 记录训练开始时间
    start_time = time.time()
    logging.info(f"Start time:{start_time}")
    
    # 遍历所有任务，进行增量训练和评估
    for task in range(data_manager.nb_tasks):
        # 记录模型的所有参数数量
        logging.info("All params: {}".format(count_parameters(model._network)))
        # 记录模型的可训练参数数量
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        
        # 执行增量训练
        model.incremental_train(data_manager)
        # 若为最后一个任务，保存评估结果的置信度
        if task == data_manager.nb_tasks-1:
            cnn_accy, nme_accy = model.eval_task(save_conf=True)
            no_nme = True if nme_accy is None else False
        else:
            cnn_accy, nme_accy = model.eval_task(save_conf=False)
        # 执行任务结束后的操作
        model.after_task()
        
        # 若存在 NME 评估结果，记录 CNN 和 NME 的准确率
        if nme_accy is not None:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top3"].append(cnn_accy["top3"])

            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve["top3"].append(nme_accy["top3"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top3 curve: {}".format(cnn_curve["top3"]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("NME top3 curve: {}\n".format(nme_curve["top3"]))
        else:
            # 若不存在 NME 评估结果，仅记录 CNN 的准确率
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy["grouped"]))

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top3"].append(cnn_accy["top3"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top3 curve: {}\n".format(cnn_curve["top3"]))
    
    # 记录训练结束时间
    end_time = time.time()
    logging.info(f"End Time:{end_time}")
    # 计算训练耗时
    cost_time = end_time - start_time
    # 保存训练耗时
    save_time(args, cost_time)
    # 保存评估结果
    save_results(args, cnn_curve, nme_curve, no_nme)
    # 根据模型名称，选择保存全连接层或整个模型
    if args['model_name'] not in ["podnet", "coil"]:
        save_fc(args, model)
    else:
        save_model(args, model)

def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))

def save_time(args, cost_time):
    _log_dir = os.path.join("./results/", "times", f"{args['prefix']}")
    os.makedirs(_log_dir, exist_ok=True)
    _log_path = os.path.join(_log_dir, f"{args['csv_name']}.csv")
    with open(_log_path, "a+") as f:
        f.write(f"{args['time_str']},{args['model_name']}, {cost_time} \n")

def save_results(args, cnn_curve, nme_curve, no_nme=False):
    cnn_top1, cnn_top3 = cnn_curve["top1"], cnn_curve['top3']
    nme_top1, nme_top3 = nme_curve["top1"], nme_curve['top3']
    
    #-------CNN TOP1----------
    _log_dir = os.path.join("./results/", f"{args['prefix']}", "cnn_top1")
    os.makedirs(_log_dir, exist_ok=True)

    _log_path = os.path.join(_log_dir, f"{args['csv_name']}.csv")
    if args['prefix'] == 'benchmark':
        with open(_log_path, "a+") as f:
            f.write(f"{args['time_str']},{args['model_name']},")
            for _acc in cnn_top1[:-1]:
                f.write(f"{_acc},")
            f.write(f"{cnn_top1[-1]} \n")
    else:
        assert args['prefix'] in ['fair', 'auc']
        with open(_log_path, "a+") as f:
            f.write(f"{args['time_str']},{args['model_name']},{args['memory_size']},")
            for _acc in cnn_top1[:-1]:
                f.write(f"{_acc},")
            f.write(f"{cnn_top1[-1]} \n")

    #-------CNN top3----------
    _log_dir = os.path.join("./results/", f"{args['prefix']}", "cnn_top3")
    os.makedirs(_log_dir, exist_ok=True)
    _log_path = os.path.join(_log_dir, f"{args['csv_name']}.csv")
    if args['prefix'] == 'benchmark':
        with open(_log_path, "a+") as f:
            f.write(f"{args['time_str']},{args['model_name']},")
            for _acc in cnn_top3[:-1]:
                f.write(f"{_acc},")
            f.write(f"{cnn_top3[-1]} \n")
    else:
        assert args['prefix'] in ['auc', 'fair']
        with open(_log_path, "a+") as f:
            f.write(f"{args['time_str']},{args['model_name']},{args['memory_size']},")
            for _acc in cnn_top3[:-1]:
                f.write(f"{_acc},")
            f.write(f"{cnn_top3[-1]} \n")


    #-------NME TOP1----------
    if no_nme is False:
        _log_dir = os.path.join("./results/", f"{args['prefix']}", "nme_top1")
        os.makedirs(_log_dir, exist_ok=True)
        _log_path = os.path.join(_log_dir, f"{args['csv_name']}.csv")
        if args['prefix'] == 'benchmark':
            with open(_log_path, "a+") as f:
                f.write(f"{args['time_str']},{args['model_name']},")
                for _acc in nme_top1[:-1]:
                    f.write(f"{_acc},")
                f.write(f"{nme_top1[-1]} \n")
        else:
            assert args['prefix'] in ['fair', 'auc']
            with open(_log_path, "a+") as f:
                f.write(f"{args['time_str']},{args['model_name']},{args['memory_size']},")
                for _acc in nme_top1[:-1]:
                    f.write(f"{_acc},")
                f.write(f"{nme_top1[-1]} \n")       

        #-------NME top3----------
        _log_dir = os.path.join("./results/", f"{args['prefix']}", "nme_top3")
        os.makedirs(_log_dir, exist_ok=True)
        _log_path = os.path.join(_log_dir, f"{args['csv_name']}.csv")
        if args['prefix'] == 'benchmark':
            with open(_log_path, "a+") as f:
                f.write(f"{args['time_str']},{args['model_name']},")
                for _acc in nme_top3[:-1]:
                    f.write(f"{_acc},")
                f.write(f"{nme_top3[-1]} \n")
        else:
            assert args['prefix'] in ['auc', 'fair']
            with open(_log_path, "a+") as f:
                f.write(f"{args['time_str']},{args['model_name']},{args['memory_size']},")
                for _acc in nme_top3[:-1]:
                    f.write(f"{_acc},")
                f.write(f"{nme_top3[-1]} \n") 
