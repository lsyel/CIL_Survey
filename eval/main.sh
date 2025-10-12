#!/bin/bash

# 主脚本：运行iCaRL和iCaRL-MoE实验

# 创建日志目录
mkdir -p logs

echo "🚀 开始执行iCaRL实验..."

# 运行iCaRL实验
/root/wzhdesign/CIL_Survey/eval/run_icarl.sh
# 检查执行结果
if [ $? -eq 0 ]; then
    echo "✅ iCaRL实验成功完成"
else
    echo "❌ iCaRL实验执行失败！"
    exit 1
fi

echo "🚀 开始执行iCaRL-MoE实验..."

# 运行iCaRL-MoE实验
/root/wzhdesign/CIL_Survey/eval/run_icarl_moe.sh
# 检查执行结果
if [ $? -eq 0 ]; then
    echo "✅ iCaRL-MoE实验成功完成"
else
    echo "❌ iCaRL-MoE实验执行失败！"
    exit 1
fi

echo "🎉 所有实验执行完毕！"