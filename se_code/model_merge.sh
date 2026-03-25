local_dir=$1
target_dir=$2

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $local_dir \
    --target_dir $target_dir