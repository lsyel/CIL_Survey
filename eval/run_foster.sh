#!/bin/bash

# 定义要测试的memory_size值
MEMORY_SIZES=( 10000 15000 )
MODEL_NAME=foster
NET=resnet34
INIT=5
INCREMENT=5
DEVICE=0

# 创建日志目录
mkdir -p eval_result


# 遍历所有memory_size值
for SIZE in "${MEMORY_SIZES[@]}"; do
    echo "===== 开始执行 memory_size=$SIZE ====="
    
    # 构建命令
    cmd="python /root/wzhdesign/CIL_Survey/main.py"
    cmd+=" --dataset ustc2016"
    cmd+=" -model $MODEL_NAME"
    cmd+=" -net $NET"
    cmd+=" -init $INIT"
    cmd+=" -incre $INCREMENT"
    cmd+=" -p benchmark"
    cmd+=" -d $DEVICE"
    cmd+=" --memory_size $SIZE"
    cmd+=" > eval_result/${MODEL_NAME}_memory_${SIZE}.log 2>&1"
    
    # 打印并执行命令
    echo "执行命令: $cmd"
    eval $cmd
    
    # 检查执行结果
    if [ $? -eq 0 ]; then
        echo "✅ ${MODEL_NAME}memory_size=$SIZE 实验成功完成"
    else
        echo "❌ ${MODEL_NAME}_memory_size=$SIZE 实验执行失败！"
        exit 1
    fi
    
    echo "===== ${MODEL_NAME}_memory_size=$SIZE 实验完成 ====="
    echo
done
