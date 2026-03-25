#!/bin/bash

# 检测GPU数量的脚本（循环检测版本）
# 如果有8个GPU，则启动main.sh

# 检测nvidia-smi是否可用
if ! command -v nvidia-smi &> /dev/null; then
    echo "错误: nvidia-smi 未找到，请确保已安装NVIDIA驱动"
    exit 1
fi

# 标记文件，用于防止重复启动main.sh
LOCK_FILE="/tmp/main_sh_running.lock"

# 循环检测
while true; do
    # 获取当前时间
    CURRENT_TIME=$(date '+%Y-%m-%d %H:%M:%S')
    
    # 获取GPU数量
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    
    echo "[$CURRENT_TIME] 检测到 GPU 数量: $GPU_COUNT"
    
    # 检查是否有8个GPU
    if [ "$GPU_COUNT" -eq 8 ]; then
        echo "[$CURRENT_TIME] GPU数量符合要求(8个)"
        
        # 检查是否已经在运行main.sh
        if [ -f "$LOCK_FILE" ]; then
            echo "[$CURRENT_TIME] main.sh 已经在运行中，跳过本次启动"
        else
            echo "[$CURRENT_TIME] 正在启动 main.sh..."
            
            # 创建锁文件
            touch "$LOCK_FILE"
            
            # 检查main.sh是否存在并启动
            if [ -f "/root/users/ycy/Self-evolving-Agent/se_code/Zero/main_gan.sh" ]; then
                nohup bash main_gan.sh > ../logs/prompt2_gan_se-Zero-$(now).log 2>&1 &
            else
                echo "[$CURRENT_TIME] 错误: 找不到 main.sh 脚本"
                rm -f "$LOCK_FILE"
            fi
            
            # main.sh执行完成后删除锁文件
            rm -f "$LOCK_FILE"
            echo "[$CURRENT_TIME] main.sh 执行完成"
        fi
    else
        echo "[$CURRENT_TIME] GPU数量不符合要求，需要8个GPU，当前有 $GPU_COUNT 个"
    fi
    
    # 等待1小时（3600秒）
    echo "[$CURRENT_TIME] 等待1小时后进行下次检测..."
    sleep 3600
done
