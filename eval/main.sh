#!/bin/bash

# 创建日志目录
mkdir -p eval_result
DIRNAME=$(dirname "$0")
find $DIRNAME -maxdepth 1 -name "run_*.sh" | while read -r script; do
    echo "🚀 开始执行 $script "
    eval $script &
    if [ $? -ne 0 ]; then
        echo "❌ $script 执行失败！"
        exit 1
    fi
done
wait
echo "🎉 所有实验执行完毕！"
