#!/usr/bin/env bash
# 资源清理函数库
# 提供多用户环境下的安全进程清理功能

# 配置参数
CLEANUP_TIMEOUT_CHECK=${CLEANUP_TIMEOUT_CHECK:-15}      # 检查残留进程超时时间（秒）
CLEANUP_TIMEOUT_FALLBACK=${CLEANUP_TIMEOUT_FALLBACK:-45} # Fallback清理超时时间（秒）
CLEANUP_PROCESS_TIMEOUT=${CLEANUP_PROCESS_TIMEOUT:-3}    # 单个进程检查超时时间（秒）
CLEANUP_SLEEP_INTERVAL=${CLEANUP_SLEEP_INTERVAL:-2}      # 进程终止后等待时间（秒）
CLEANUP_PORTS=${CLEANUP_PORTS:-"5000 5001 5002 5003"}   # 需要清理的端口列表

# 获取当前会话的唯一标识符
get_session_id() {
    local USER_ID=$(whoami)
    local SESSION_ID="$$"
    local PROCESS_GROUP_ID=$(ps -o pgid= -p $$ | tr -d ' ')
    local SESSION_PID=$(ps -o sid= -p $$ | tr -d ' ')
    local UNIQUE_SESSION_ID="${USER_ID}_${SESSION_PID}_${PROCESS_GROUP_ID}_${SESSION_ID}"
    
    echo "$UNIQUE_SESSION_ID"
}

# 获取进程跟踪文件路径
# 参数: $1 - 脚本名称前缀 (可选，默认为 "solver")
get_process_track_file() {
    local script_prefix="${1:-solver}"  # 默认使用 "solver" 作为前缀
    local tmp="${2:-/tmp}"              # 默认使用 "/tmp"，但可以自定义
    local session_id=$(get_session_id)
    echo "${tmp}/${script_prefix}_processes_${session_id}.txt"
}

# 检查进程是否属于当前用户会话
is_my_process() {
    local pid=$1
    if [ -z "$pid" ]; then
        return 1
    fi
    
    # 检查进程是否存在
    if ! ps -p "$pid" > /dev/null 2>&1; then
        return 1
    fi
    
    # 获取当前会话信息
    local SESSION_ID="$$"
    local PROCESS_GROUP_ID=$(ps -o pgid= -p $$ | tr -d ' ')
    local SESSION_PID=$(ps -o sid= -p $$ | tr -d ' ')
    
    # 获取进程的会话ID和进程组ID
    local proc_sid=$(ps -o sid= -p "$pid" 2>/dev/null | tr -d ' ')
    local proc_pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
    
    # 检查是否属于当前会话或进程组
    if [ "$proc_sid" = "$SESSION_PID" ] || [ "$proc_pgid" = "$PROCESS_GROUP_ID" ]; then
        return 0
    fi
    
    # 额外检查：如果是我们启动的进程，检查其父进程链
    local ppid=$pid
    local max_depth=10
    local depth=0
    
    while [ "$ppid" != "1" ] && [ "$depth" -lt "$max_depth" ]; do
        if [ "$ppid" = "$SESSION_ID" ] || [ "$ppid" = "$PROCESS_GROUP_ID" ]; then
            return 0
        fi
        ppid=$(ps -o ppid= -p "$ppid" 2>/dev/null | tr -d ' ')
        depth=$((depth + 1))
    done
    
    return 1
}

# 获取属于当前用户的进程列表
get_my_processes() {
    local pattern=$1
    local result=""
    while read -r pid; do
        if is_my_process "$pid"; then
            result="$result $pid"
        fi
    done < <(pgrep -f "$pattern" 2>/dev/null || true)
    echo "$result"
}

# 进程跟踪函数
# 参数: $1 - PID, $2 - 描述, $3 - 脚本前缀 (可选), $4 - tmp路径 (可选)
track_process() {
    local pid=$1
    local description=$2
    local script_prefix="${3:-solver}"  # 默认使用 "solver" 作为前缀
    local tmp="${4:-/tmp}"              # 默认使用 "/tmp"，但可以自定义
    local track_file=$(get_process_track_file "$script_prefix" "$tmp")
    echo "$pid:$description" >> "$track_file"
    echo "跟踪进程: PID=$pid, 描述=$description"
}

# 显示跟踪的进程状态
# 参数: $1 - 脚本前缀 (可选), $2 - tmp路径 (可选)
show_tracked_processes() {
    local script_prefix="${1:-solver}"  # 默认使用 "solver" 作为前缀
    local tmp="${2:-/tmp}"              # 默认使用 "/tmp"，但可以自定义
    local track_file=$(get_process_track_file "$script_prefix" "$tmp")
    if [ -f "$track_file" ]; then
        echo "当前跟踪的进程 (${script_prefix}):"
        while IFS=':' read -r pid description; do
            if [ -n "$pid" ] && [ -n "$description" ]; then
                if ps -p "$pid" > /dev/null 2>&1; then
                    echo "  ✓ PID $pid ($description) - 运行中"
                else
                    echo "  ✗ PID $pid ($description) - 已结束"
                fi
            fi
        done < "$track_file"
    else
        echo "没有跟踪的进程 (${script_prefix})"
    fi
}

# 清理跟踪的进程
# 参数: $1 - 脚本前缀 (可选), $2 - tmp路径 (可选)
cleanup_tracked_processes() {
    local script_prefix="${1:-solver}"  # 默认使用 "solver" 作为前缀
    local tmp="${2:-/tmp}"              # 默认使用 "/tmp"，但可以自定义
    local track_file=$(get_process_track_file "$script_prefix" "$tmp")
    local tracked_cleaned=false
    
    if [ -f "$track_file" ]; then
        echo "第一阶段：清理跟踪的进程..."
        while IFS=':' read -r pid description; do
            if [ -n "$pid" ] && [ -n "$description" ]; then
                if ps -p "$pid" > /dev/null 2>&1; then
                    echo "  终止进程: PID=$pid ($description)"
                    recursive_kill_process "$pid" "TERM" "$description"
                    sleep 2
                    if ps -p "$pid" > /dev/null 2>&1; then
                        echo "  强制终止进程: PID=$pid"
                        recursive_kill_process "$pid" "KILL" "$description"
                    fi
                    tracked_cleaned=true
                fi
            fi
        done < "$track_file"
        rm -f "$track_file"
        echo "跟踪进程清理完成"
    fi
    
    return $([ "$tracked_cleaned" = true ] && echo 0 || echo 1)
}

# 带超时的进程检查函数
check_processes_with_timeout() {
    local pattern=$1
    local timeout=${2:-$CLEANUP_PROCESS_TIMEOUT}  # 使用配置的超时时间
    local result=""
    
    # 使用timeout命令限制执行时间
    result=$(timeout $timeout bash -c "
        local pids=\$(pgrep -f \"$pattern\" 2>/dev/null || true)
        if [ -n \"\$pids\" ]; then
            for pid in \$pids; do
                if is_my_process \"\$pid\"; then
                    echo \"\$pid\"
                fi
            done
        fi
    " 2>/dev/null || echo "")
    
    echo "$result"
}

# 递归清理进程及其所有子进程（简化版本）
recursive_kill_process() {
    local target_pid="$1"
    local signal="${2:-TERM}"
    local description="${3:-进程}"
    
    if ! ps -p "$target_pid" > /dev/null 2>&1; then
        return 0  # 进程不存在，直接返回
    fi
    
    # 获取进程信息
    local cmd=$(ps -p "$target_pid" -o cmd= 2>/dev/null || echo "unknown")
    echo "    终止${description}: PID=$target_pid, CMD=$cmd"
    
    # 使用pkill -P 来递归终止所有子进程
    # 这会先终止子进程，再终止父进程
    pkill -P "$target_pid" -"$signal" 2>/dev/null || true
    
    # 给子进程一些时间退出
    sleep 0.5
    
    # 最后发送信号给父进程
    kill -"$signal" "$target_pid" 2>/dev/null || true
}

# 检查残留进程（优化版本）
check_residual_processes() {
    local need_fallback=false
    
    echo "  检查残留进程（带超时保护）..."
    
    # 检查Ray进程
    local ray_pids=$(check_processes_with_timeout "ray::")
    if [ -n "$ray_pids" ]; then
        echo "  发现残留Ray进程: $ray_pids"
        need_fallback=true
    fi
    
    # 检查训练进程
    local training_pids=$(check_processes_with_timeout "python3 -m se.main_solver_dapo")
    if [ -n "$training_pids" ]; then
        echo "  发现残留训练进程: $training_pids"
        need_fallback=true
    fi
    
    # 检查challenger训练进程
    local challenger_pids=$(check_processes_with_timeout "python3 -m se.main_challenger")
    if [ -n "$challenger_pids" ]; then
        echo "  发现残留challenger训练进程: $challenger_pids"
        need_fallback=true
    fi
    
    # 检查模型合并进程
    local merge_pids=$(check_processes_with_timeout "python3 -m verl.model_merger")
    if [ -n "$merge_pids" ]; then
        echo "  发现残留模型合并进程: $merge_pids"
        need_fallback=true
    fi
    
    # 检查vLLM进程（使用更精确的模式）
    local vllm_pids=$(check_processes_with_timeout "start_vllm_server.py")
    if [ -n "$vllm_pids" ]; then
        echo "  发现残留vLLM服务器进程: $vllm_pids"
        need_fallback=true
    fi
    
    # 检查端口占用（强制检查所有进程）
    local port_pids=""
    for port in $CLEANUP_PORTS; do
        local port_users=$(lsof -ti:$port 2>/dev/null || true)
        if [ -n "$port_users" ]; then
            for pid in $port_users; do
                # 检查进程是否存在
                if ps -p "$pid" > /dev/null 2>&1; then
                    local user=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
                    local cmd=$(ps -p "$pid" -o cmd= 2>/dev/null | head -c 100)
                    echo "    发现端口${port}占用: PID=$pid, USER=$user, CMD=$cmd"
                    port_pids="$port_pids $pid"
                fi
            done
        fi
    done
    if [ -n "$port_pids" ]; then
        echo "  发现端口占用进程: $port_pids"
        need_fallback=true
    fi
    
    echo "$need_fallback"
}

# 强制清理所有vLLM相关进程（整合force_cleanup_vllm.sh的逻辑）
force_cleanup_vllm_processes() {
    echo "  强制清理所有vLLM相关进程..."
    
    # 显示清理前的进程状态
    echo "    清理前的vLLM进程状态:"
    ps aux | grep start_vllm_server.py | grep -v grep || echo "      没有发现vLLM进程"
    
    # 1. 获取所有start_vllm_server.py进程的PID
    local vllm_pids=$(ps aux | grep start_vllm_server.py | grep -v grep | awk '{print $2}' 2>/dev/null || true)
    
    if [ -n "$vllm_pids" ]; then
        echo "    发现以下vLLM进程: $vllm_pids"
        
        # 发送TERM信号
        echo "    发送TERM信号..."
        for pid in $vllm_pids; do
            echo "      终止进程: PID=$pid"
            kill -TERM "$pid" 2>/dev/null || true
        done
        
        # 等待进程退出
        echo "    等待进程退出..."
        sleep 3
        
        # 检查是否还有进程存在
        local remaining_pids=$(ps aux | grep start_vllm_server.py | grep -v grep | awk '{print $2}' 2>/dev/null || true)
        if [ -n "$remaining_pids" ]; then
            echo "    仍有进程未退出，发送KILL信号..."
            for pid in $remaining_pids; do
                echo "      强制终止进程: PID=$pid"
                kill -KILL "$pid" 2>/dev/null || true
            done
            sleep 1
        fi
    else
        echo "    没有发现需要清理的vLLM进程"
    fi
    
    # 2. 清理vLLM多进程子进程（orphaned processes）
    echo "    清理vLLM多进程子进程..."
    local vllm_orphan_pids=$(ps aux | grep "multiprocessing.spawn import spawn_main" | grep -v grep | awk '{print $2}' 2>/dev/null || true)
    if [ -n "$vllm_orphan_pids" ]; then
        echo "    发现vLLM子进程: $vllm_orphan_pids"
        for pid in $vllm_orphan_pids; do
            echo "      终止vLLM子进程: PID=$pid"
            kill -TERM "$pid" 2>/dev/null || true
        done
        sleep 2
        # 检查是否还有残留进程
        local remaining_orphan_pids=$(ps aux | grep "multiprocessing.spawn import spawn_main" | grep -v grep | awk '{print $2}' 2>/dev/null || true)
        if [ -n "$remaining_orphan_pids" ]; then
            echo "    强制终止残留vLLM子进程..."
            for pid in $remaining_orphan_pids; do
                echo "      强制终止vLLM子进程: PID=$pid"
                kill -KILL "$pid" 2>/dev/null || true
            done
        fi
    else
        echo "    没有发现需要清理的vLLM子进程"
    fi
    
    # 显示清理后的进程状态
    echo "    清理后的vLLM进程状态:"
    ps aux | grep start_vllm_server.py | grep -v grep || echo "      没有发现vLLM进程"
    ps aux | grep "multiprocessing.spawn import spawn_main" | grep -v grep || echo "      没有发现vLLM子进程"
    
    # 显示清理后的端口占用状态
    echo "    清理后的端口占用状态:"
    for port in $CLEANUP_PORTS; do
        local port_pids=$(lsof -ti:$port 2>/dev/null || true)
        if [ -n "$port_pids" ]; then
            echo "      端口 $port 仍被进程占用: $port_pids"
        else
            echo "      端口 $port 空闲"
        fi
    done
    
    echo "  vLLM进程强制清理完成"
}

# 统一清理vLLM相关进程（保持向后兼容）
cleanup_vllm_processes() {
    force_cleanup_vllm_processes
}

# 基于端口清理vLLM进程（简单可靠的方法）
cleanup_vllm_ports() {
    echo "清理vLLM端口进程..."
    for port in $CLEANUP_PORTS; do
        local port_pids=$(lsof -ti:$port 2>/dev/null || true)
        if [ -n "$port_pids" ]; then
            echo "  清理端口 $port 上的进程（递归）: $port_pids"
            for pid in $port_pids; do
                echo "    终止端口 $port 进程: PID=$pid"
                recursive_kill_process "$pid" "TERM" "端口${port}进程"
            done
            sleep 1
            for pid in $port_pids; do
                recursive_kill_process "$pid" "KILL" "端口${port}进程"
            done
        else
            echo "  端口 $port: 空闲"
        fi
    done
    echo "vLLM端口清理完成"
}

# Fallback清理残留进程（优化版本）
cleanup_residual_processes() {
    echo "第三阶段：执行fallback清理..."
    
    # 清理残留Ray进程（递归清理）
    local ray_pids=$(check_processes_with_timeout "ray::")
    if [ -n "$ray_pids" ]; then
        echo "  清理残留Ray进程..."
        for pid in $ray_pids; do
            recursive_kill_process "$pid" "TERM" "Ray进程"
        done
        sleep 1
        for pid in $ray_pids; do
            recursive_kill_process "$pid" "KILL" "Ray进程"
        done
    fi
    
    # 清理残留训练进程（递归清理）
    local training_pids=$(check_processes_with_timeout "python3 -m se.main_solver_dapo")
    if [ -n "$training_pids" ]; then
        echo "  清理残留训练进程..."
        for pid in $training_pids; do
            recursive_kill_process "$pid" "TERM" "训练进程"
        done
        sleep 1
        for pid in $training_pids; do
            recursive_kill_process "$pid" "KILL" "训练进程"
        done
    fi
    
    # 清理残留challenger训练进程（递归清理）
    local challenger_pids=$(check_processes_with_timeout "python3 -m se.main_challenger")
    if [ -n "$challenger_pids" ]; then
        echo "  清理残留challenger训练进程..."
        for pid in $challenger_pids; do
            recursive_kill_process "$pid" "TERM" "challenger训练进程"
        done
        sleep 1
        for pid in $challenger_pids; do
            recursive_kill_process "$pid" "KILL" "challenger训练进程"
        done
    fi
    
    # 清理残留模型合并进程（递归清理）
    local merge_pids=$(check_processes_with_timeout "python3 -m verl.model_merger")
    if [ -n "$merge_pids" ]; then
        echo "  清理残留模型合并进程..."
        for pid in $merge_pids; do
            recursive_kill_process "$pid" "TERM" "模型合并进程"
        done
        sleep 1
        for pid in $merge_pids; do
            recursive_kill_process "$pid" "KILL" "模型合并进程"
        done
    fi
    
    # 统一清理vLLM相关进程
    cleanup_vllm_processes
    
    # 强制清理端口占用进程（递归清理，检查用户权限）
    echo "  强制清理端口占用进程..."
    for port in $CLEANUP_PORTS; do
        local port_users=$(lsof -ti:$port 2>/dev/null || true)
        if [ -n "$port_users" ]; then
            echo "    清理端口${port}占用进程: $port_users"
            for pid in $port_users; do
                if ps -p "$pid" > /dev/null 2>&1; then
                    # 检查进程是否属于当前用户
                    local proc_user=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
                    local current_user=$(whoami)
                    if [ "$proc_user" = "$current_user" ]; then
                        recursive_kill_process "$pid" "KILL" "端口${port}占用进程"
                    else
                        echo "      跳过其他用户进程: PID=$pid, USER=$proc_user"
                    fi
                fi
            done
        fi
    done
    
    echo "Fallback清理完成"
}

# 显示GPU状态
show_gpu_status() {
    echo "当前GPU状态:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo "无法获取GPU状态"
    echo ""
}

# 主清理函数
# 参数: $1 - 脚本前缀 (可选), $2 - tmp路径 (可选)
cleanup_processes() {
    local script_prefix="${1:-solver}"  # 默认使用 "solver" 作为前缀
    local tmp="${2:-/tmp}"              # 默认使用 "/tmp"，但可以自定义
    set +e
    echo "=========================================="
    echo "开始清理相关进程 (${script_prefix})..."
    
    # 显示当前GPU状态
    show_gpu_status
    
    # 显示跟踪的进程状态
    echo "清理前进程状态:"
    show_tracked_processes "$script_prefix" "$tmp"
    echo ""
    
    # 第一阶段：清理跟踪的进程
    cleanup_tracked_processes "$script_prefix" "$tmp"
    
    # 等待进程退出
    sleep 2
    
    # 第二阶段：检查残留进程（带超时保护）
    echo "第二阶段：检查残留进程..."
    local need_fallback
    need_fallback=$(timeout $CLEANUP_TIMEOUT_CHECK check_residual_processes 2>/dev/null || echo "true")
    
    # 第三阶段：Fallback清理（带超时保护）
    if [ "$need_fallback" = true ]; then
        echo "执行fallback清理（带超时保护）..."
        if timeout $CLEANUP_TIMEOUT_FALLBACK cleanup_residual_processes 2>/dev/null; then
            echo "Fallback清理完成"
        else
            echo "警告：Fallback清理超时，但已尽力清理"
        fi
    else
        echo "没有发现残留进程，跳过fallback清理"
    fi
    
    # 显示清理后的GPU状态
    echo ""
    echo "清理后GPU状态:"
    show_gpu_status
    
    echo "进程清理完成"
    echo "=========================================="
}





# 检查是否禁用进程清理
check_cleanup_option() {
    local arg_count="$1"
    local last_arg="$2"
    
    if [ $arg_count -eq 5 ] && [ "$last_arg" = "--no-cleanup" ]; then
        echo "false"
    else
        echo "true"
    fi
}




# 独立的强制清理vLLM函数（可以直接调用）
force_cleanup_vllm() {
    echo "=========================================="
    echo "强制清理所有start_vllm_server.py进程"
    echo "=========================================="
    
    force_cleanup_vllm_processes
    
    echo ""
    echo "=========================================="
    echo "强制清理完成"
    echo "=========================================="
}

# 如果直接执行此脚本，则执行清理
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    # 检查是否指定了强制清理vLLM
    if [ "$1" = "--force-vllm" ]; then
        force_cleanup_vllm
    else
        cleanup_processes "${1:-solver}"
    fi
fi

