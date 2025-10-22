#!/bin/bash

# 定义要测试的memory_size值
MEMORY_SIZES=( 10000  )

# 创建日志目录
mkdir -p eval_result


# 遍历所有memory_size值
for SIZE in "${MEMORY_SIZES[@]}"; do
    echo "===== 开始执行 memory_size=$SIZE ====="
    
    # 构建命令
    cmd="python /root/wzhdesign/CIL_Survey/main.py"
    cmd+=" --dataset ustc2016"
    cmd+=" -model icarl"
    cmd+=" -net my_resnet34"
    cmd+=" -init 5"
    cmd+=" -incre 5"
    cmd+=" -p benchmark"
    cmd+=" -d 1"
    cmd+=" --memory_size $SIZE"
    cmd+=" > eval_result/icarl_memory_${SIZE}.log 2>&1"
    
    # 打印并执行命令
    echo "执行命令: $cmd"
    eval $cmd
    
    # 检查执行结果
    if [ $? -eq 0 ]; then
        echo "✅ memory_size=$SIZE 实验成功完成"
    else
        echo "❌ memory_size=$SIZE 实验执行失败！"
        exit 1
    fi
    
    echo "===== memory_size=$SIZE 实验完成 ====="
    echo
done
