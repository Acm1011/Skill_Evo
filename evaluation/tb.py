from verl.utils.tracking import Tracking
import os
import pandas as pd
import json
from collections import OrderedDict


def read_eval(path, greedy=False):
    res = {}
    eval_data=pd.read_json(path, lines=True).to_dict(orient='records')
    all_data=[]
    for data_item in eval_data:
        assert isinstance(data_item,dict), f'{type(data_item)=}'
        data_source=data_item['data_source']
        if greedy:
            data=data_item['checked_mean@1']
            res.update({
                f'checked_mean@1/{data_source}':data
            })

        else:
            data=data_item['checked_mean@32']
            res.update({
                f'checked_mean@32/{data_source}':data
            })
        all_data.append(data)
    
    return res, all_data


def read_aggregated_eval(eval_path):
    """读取三个新数据集的聚合评测结果
    
    Returns:
        res: 包含各个数据集结果的字典
        general_avg_data: 三个数据集的准确率列表，用于计算 General AVG
    """
    res = {}
    general_avg_data = []
    aggregated_path = os.path.join(eval_path, 'aggregated_eval_results.json')
    
    if not os.path.exists(aggregated_path):
        print(f"Warning: Aggregated eval results not found: {aggregated_path}")
        return res, general_avg_data
    
    try:
        with open(aggregated_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            datasets = data.get('datasets', {})
            
            # 提取通用评测数据集的结果
            for dataset_name in ['bbeh', 'mmlupro', 'supergpqa', 'gpqa']:
                if dataset_name in datasets:
                    dataset_result = datasets[dataset_name]
                    accuracy = dataset_result.get('accuracy')
                    #macro_accuracy = dataset_result.get('macro_accuracy')
                    
                    if accuracy is not None:
                        # 将准确率转换为百分比（如果已经是百分比则保持，如果是小数则转换）
                        if accuracy <= 1.0:
                            accuracy = accuracy * 100
                        res[f'additional_eval/{dataset_name}_accuracy'] = round(accuracy, 2)
                        general_avg_data.append(accuracy)
                    
                    # if macro_accuracy is not None:
                    #     if macro_accuracy <= 1.0:
                    #         macro_accuracy = macro_accuracy * 100
                    #     res[f'additional_eval/{dataset_name}_macro_accuracy'] = round(macro_accuracy, 2)
    except Exception as e:
        print(f"Error reading aggregated eval results: {e}")
    
    return res, general_avg_data
    

def tb(prefix, step, temperature, eval_results_dir=None, tb_path_dir=None, base_model=None, base_model_dir=None, step_interval=20):
    if tb_path_dir is None:
        raise ValueError(f'{tb_path_dir=} is None')
    os.makedirs(tb_path_dir,exist_ok=True)
    if eval_results_dir is None:
        raise ValueError(f'{eval_results_dir=} is None')
    
    # 计算模型迭代次数：只统计符合 {prefix}-V{数字} 格式的目录
    n_iter = 0
    import re
    pattern = re.compile(rf'^{re.escape(prefix)}-V(\d+)$')
    for path in os.listdir(eval_results_dir):
        full_path = os.path.join(eval_results_dir, path)
        if os.path.isdir(full_path) and pattern.match(path):
            match = pattern.match(path)
            version = int(match.group(1))
            if version > n_iter:
                n_iter = version
    
    tb_path=os.path.join(tb_path_dir, f'{prefix}-step_{step}-temperature_{temperature}')
    os.environ['TENSORBOARD_DIR']=tb_path
    logger=Tracking(project_name='Se', experiment_name=prefix, default_backend=['console','tensorboard'])

    # base_model 必须提供
    if not base_model:
        raise ValueError("--base_model is required. Please specify the base model name (e.g., Qwen2.5-Math-1.5B)")
    
    for i in range(0, n_iter+1):
        if i == 0:
            suff = base_model
            # base_model_dir 用于指定基础模型结果的目录（可能与 eval_results_dir 不同）
            if base_model_dir:
                eval_path = os.path.join(base_model_dir, suff)
            else:
                eval_path = os.path.join(eval_results_dir, suff)
        else:
            suff=f'{prefix}-V{i}'
            eval_path=os.path.join(eval_results_dir, suff)
            
        if not os.path.exists(eval_path):
            raise ValueError(f'{eval_path=} not exists')
        eval_data = {}
        greedy_data_path=os.path.join(eval_path, 'greedy_data_Overall_results.jsonl')
        tmp_data_path=os.path.join(eval_path, 'temp_data_Overall_results.jsonl')
        
        # 读取数学数据集结果，处理文件不存在的情况
        d1 = {}
        d2 = {}
        avg_d1 = []
        avg_d2 = []
        
        if os.path.exists(greedy_data_path):
            d1, avg_d1 = read_eval(greedy_data_path, greedy=True)
        else:
            print(f"Warning: {greedy_data_path} not found")
        
        if os.path.exists(tmp_data_path):
            d2, avg_d2 = read_eval(tmp_data_path, greedy=False)
        else:
            print(f"Warning: {tmp_data_path} not found")
        
        # 计算 Math AVG（原来的数学数据集平均值）
        math_avg_data = avg_d1 + avg_d2
        math_avg = sum(math_avg_data) / len(math_avg_data) if len(math_avg_data) > 0 else 0.0
        
        eval_data.update(d1)
        eval_data.update(d2)
        eval_data.update({"Math_AVG": round(math_avg, 2)})
        
        # 添加通用数据集的评测结果
        additional_eval_data, general_avg_data = read_aggregated_eval(eval_path)
        eval_data.update(additional_eval_data)
        
        # 计算 General AVG（通用数据集的平均值）
        general_avg = sum(general_avg_data) / len(general_avg_data) if len(general_avg_data) > 0 else 0.0
        eval_data.update({"General_AVG": round(general_avg, 2)})
        
        # 计算 Overall AVG（所有数据集的总平均值）
        all_avg_data = math_avg_data + general_avg_data
        overall_avg = sum(all_avg_data) / len(all_avg_data) if len(all_avg_data) > 0 else 0.0
        eval_data.update({"Overall_AVG": round(overall_avg, 2)})
        
        # 使用 step_interval 计算实际的 step 值：V1 = step_interval, V2 = 2*step_interval, etc.
        actual_step = i * step_interval
        logger.log(data=eval_data, step=actual_step)


def generate_results_table(prefix, step, temperature, output_file=None, eval_results_dir=None, base_model=None, base_model_dir=None):
    """生成结果表格，横轴是数据集和avg，纵轴是模型名字"""
    if eval_results_dir is None:
        eval_results_dir='/home/ycy/sdi/saved_results/evaluation'
    
    # 计算模型迭代次数：只统计符合 {prefix}-V{数字} 格式的目录
    import re
    n_iter = 0
    pattern = re.compile(rf'^{re.escape(prefix)}-V(\d+)$')
    for path in os.listdir(eval_results_dir):
        full_path = os.path.join(eval_results_dir, path)
        if os.path.isdir(full_path) and pattern.match(path):
            match = pattern.match(path)
            version = int(match.group(1))
            if version > n_iter:
                n_iter = version
    
    # base_model 必须提供
    if not base_model:
        raise ValueError("--base_model is required. Please specify the base model name (e.g., Qwen2.5-Math-1.5B)")
    base_model_name = base_model
    
    # 收集所有数据集名称（用于确定列顺序）
    math_datasets_set = set()  # 用于去重
    all_general_datasets = ['bbeh', 'mmlupro', 'supergpqa']  # 已排序
    
    # 先遍历一次，收集所有数学数据集名称
    for i in range(0, n_iter+1):
        if i == 0:
            suff = base_model_name
            if base_model_dir:
                eval_path = os.path.join(base_model_dir, suff)
            else:
                eval_path = os.path.join(eval_results_dir, suff)
        else:
            suff=f'{prefix}-V{i}'
            eval_path=os.path.join(eval_results_dir, suff)
        if not os.path.exists(eval_path):
            continue
        
        greedy_data_path=os.path.join(eval_path, 'greedy_data_Overall_results.jsonl')
        tmp_data_path=os.path.join(eval_path, 'temp_data_Overall_results.jsonl')
        
        if os.path.exists(greedy_data_path):
            eval_data=pd.read_json(greedy_data_path, lines=True).to_dict(orient='records')
            for data_item in eval_data:
                data_source = data_item.get('data_source')
                if data_source:
                    math_datasets_set.add((1, data_source))  # (type, name) type: 1=greedy, 2=temp
        
        if os.path.exists(tmp_data_path):
            eval_data=pd.read_json(tmp_data_path, lines=True).to_dict(orient='records')
            for data_item in eval_data:
                data_source = data_item.get('data_source')
                if data_source:
                    math_datasets_set.add((2, data_source))
    
    # 对数学数据集进行排序：先按 data_source 排序，再按 type 排序（greedy 在前）
    sorted_math_datasets = sorted(math_datasets_set, key=lambda x: (x[1], x[0]))
    all_math_datasets = []
    for dataset_type, data_source in sorted_math_datasets:
        if dataset_type == 1:
            all_math_datasets.append(f'checked_mean@1/{data_source}')
        else:
            all_math_datasets.append(f'checked_mean@32/{data_source}')
    
    # 构建表格数据
    table_data = []
    model_names = []
    
    for i in range(0, n_iter+1):
        if i == 0:
            model_name = base_model_name
            suff = base_model_name
            if base_model_dir:
                eval_path = os.path.join(base_model_dir, suff)
            else:
                eval_path = os.path.join(eval_results_dir, suff)
        else:
            model_name = f'{prefix}-V{i}'
            suff = model_name
            eval_path = os.path.join(eval_results_dir, suff)
        
        if not os.path.exists(eval_path):
            continue
        
        row = OrderedDict()
        row['Model'] = model_name
        model_names.append(model_name)
        
        # 读取数学数据集结果
        greedy_data_path=os.path.join(eval_path, 'greedy_data_Overall_results.jsonl')
        tmp_data_path=os.path.join(eval_path, 'temp_data_Overall_results.jsonl')
        
        math_values = []
        d1 = {}
        d2 = {}
        
        if os.path.exists(greedy_data_path):
            d1, avg_d1 = read_eval(greedy_data_path, greedy=True)
        
        if os.path.exists(tmp_data_path):
            d2, avg_d2 = read_eval(tmp_data_path, greedy=False)
        
        # 按顺序添加数学数据集结果（使用简短的数据集名称作为列名）
        for key in all_math_datasets:
            # 提取数据集名称（去掉 checked_mean@x/ 前缀）
            dataset_name = key.split('/')[-1]
            if key.startswith('checked_mean@1/'):
                value = d1.get(key, None)
                row[dataset_name] = round(value, 2) if value is not None else None
                if value is not None:
                    math_values.append(value)
            elif key.startswith('checked_mean@32/'):
                value = d2.get(key, None)
                row[dataset_name] = round(value, 2) if value is not None else None
                if value is not None:
                    math_values.append(value)
        
        # 计算 Math AVG
        math_avg = sum(math_values) / len(math_values) if len(math_values) > 0 else None
        row['Math_AVG'] = round(math_avg, 2) if math_avg is not None else None
        
        # 读取 General 数据集结果
        additional_eval_data, general_avg_data = read_aggregated_eval(eval_path)
        
        for dataset_name in all_general_datasets:
            key = f'additional_eval/{dataset_name}_accuracy'
            value = additional_eval_data.get(key, None)
            # 使用 dataset_name 作为列名（而不是完整的 key）
            row[dataset_name] = round(value, 2) if value is not None else None
        
        # 计算 General AVG
        general_avg = sum(general_avg_data) / len(general_avg_data) if len(general_avg_data) > 0 else None
        row['General_AVG'] = round(general_avg, 2) if general_avg is not None else None
        
        # 计算 Overall AVG
        all_values = math_values + general_avg_data
        overall_avg = sum(all_values) / len(all_values) if len(all_values) > 0 else None
        row['Overall_AVG'] = round(overall_avg, 2) if overall_avg is not None else None
        
        table_data.append(row)
    
    # 创建 DataFrame
    df = pd.DataFrame(table_data)
    
    # 确保列的顺序：Model -> 数学数据集（排序） -> Math_AVG -> General 数据集（排序） -> General_AVG -> Overall_AVG
    column_order = ['Model']
    # 使用简短的数据集名称（去掉 checked_mean@x/ 前缀）
    math_dataset_names = [key.split('/')[-1] for key in all_math_datasets]
    column_order.extend(math_dataset_names)
    column_order.append('Math_AVG')
    column_order.extend(all_general_datasets)
    column_order.append('General_AVG')
    column_order.append('Overall_AVG')
    
    # 只保留存在的列
    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order]
    
    # 确保行的顺序：Base (iter 0) 在最前面，然后按 V1, V2, ... 排序
    def sort_key(model_name):
        if model_name == base_model_name:
            return (0, '')
        elif '-V' in model_name:
            try:
                version = int(model_name.split('-V')[-1])
                return (1, version)
            except:
                return (2, model_name)
        else:
            return (2, model_name)
    
    df['_sort_key'] = df['Model'].apply(sort_key)
    df = df.sort_values('_sort_key')
    df = df.drop('_sort_key', axis=1)
    df = df.reset_index(drop=True)
    
    # 设置输出文件路径
    if output_file is None:
        output_file = os.path.join(eval_results_dir, f'{prefix}_results_table.csv')
    
    # 保存为 CSV
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Results table saved to: {output_file}")
    
    # 同时保存为 Markdown 表格（更易读）
    md_file = output_file.replace('.csv', '.md')
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(f"# Evaluation Results Table\n\n")
        f.write(f"Prefix: {prefix}, Step: {step}, Temperature: {temperature}\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    print(f"Results table (Markdown) saved to: {md_file}")
    
    return df


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='eval')
    parser.add_argument(
        "--prefix",
        type=str,
        default="prompt2_se-Zero_Qwen3-4B-Base",
        help="Prefix string for tb function."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Temperature for tb function."
    )
    parser.add_argument(
        "--step",
        type=int,
        default=15,
        help="Step number for tb function."
    )
    parser.add_argument(
        "--generate_table",
        action="store_true",
        default=True,
        help="Generate results table after uploading to tensorboard."
    )
    parser.add_argument(
        "--table_output",
        type=str,
        default=None,
        help="Output file path for results table (optional)."
    )
    parser.add_argument(
        "--eval_results_dir",
        type=str,
        default=None,
        help="Directory containing evaluation results."
    )
    parser.add_argument(
        "--tb_path_dir",
        type=str,
        default=None,
        help="Directory containing tensorboard logs."
    )
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Base model name for step 0 (e.g., Qwen2.5-Math-1.5B). Required."
    )
    parser.add_argument(
        "--base_model_dir",
        type=str,
        default=None,
        help="Directory containing base model evaluation results (if different from eval_results_dir)."
    )
    parser.add_argument(
        "--step_interval",
        type=int,
        default=20,
        help="Step interval for each version (V1 = step_interval, V2 = 2*step_interval, etc.). Default: 20."
    )
    args = parser.parse_args()
    tb(args.prefix, args.step, args.temperature, args.eval_results_dir, args.tb_path_dir, 
       args.base_model, args.base_model_dir, args.step_interval)
    
    if args.generate_table:
        generate_results_table(args.prefix, args.step, args.temperature, args.table_output, args.eval_results_dir,
                               args.base_model, args.base_model_dir)
