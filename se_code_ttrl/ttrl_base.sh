model_list=(
    Qwen2.5-Math-7B
)
train_data_list=(
    AIME24
    AIME25
    AMC23
    MATH500
    Minerva
    OlympiadBench
)
for model_name in ${model_list[@]}; do
    for train_data in ${train_data_list[@]}; do
        bash solver_base_ttrl.sh ${train_data} ${model_name}
        pkill -f python
    done
done