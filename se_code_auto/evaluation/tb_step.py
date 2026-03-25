from verl.utils.tracking import Tracking
import os
import pandas as pd


def read_eval(path, greedy=False):
    """
    读取单个评测结果文件
    
    Args:
        path: 评测结果文件路径
        greedy: 是否为greedy模式（n_samples=1）
    
    Returns:
        res: 字典，包含各个数据源的评测结果
        all_data: 列表，包含所有数据源的评测分数（用于计算平均值）
    """
    res = {}
    eval_data=pd.read_json(path, lines=True).to_dict(orient='records')
    all_data=[]
    for data_item in eval_data:
        assert isinstance(data_item,dict), f'{type(data_item)=}'
        data_source=data_item['data_source']
        if greedy:
            # greedy模式使用n_samples=1的结果
            data=data_item['checked_mean@1']
            res.update({
                f'checked_mean@1/{data_source}':data
            })
        else:
            # temp模式使用n_samples=32的结果
            data=data_item['checked_mean@32']
            res.update({
                f'checked_mean@32/{data_source}':data
            })
        all_data.append(data)
    
    return res, all_data
    

def tb_step(model_name, step_list=None):
    """
    按step上传评测结果到tensorboard
    
    Args:
        model_name: 模型名称，例如 'filter_data_Qwen3-4B-Base'
        step_list: 要上传的step列表，如果为None则自动检测所有可用的step
    """
    tb_path_dir='/root/users/ycy/saved_results/eval_tb_log'
    os.makedirs(tb_path_dir, exist_ok=True)
    eval_results_dir='/root/users/ycy/saved_results/evaluation'
    
    # 设置tensorboard日志路径
    tb_path=os.path.join(tb_path_dir, f'{model_name}')
    os.environ['TENSORBOARD_DIR']=tb_path
    logger=Tracking(project_name='Se', experiment_name=model_name, default_backend=['console','tensorboard'])
    
    # 模型结果目录
    model_eval_dir = os.path.join(eval_results_dir, model_name)
    
    if not os.path.exists(model_eval_dir):
        raise ValueError(f'Model evaluation directory not found: {model_eval_dir}')
    
    # 如果没有指定step_list，自动检测所有可用的step
    if step_list is None:
        step_list = []
        for item in os.listdir(model_eval_dir):
            if item.startswith('step_'):
                try:
                    step_num = int(item.split('_')[1])
                    step_list.append(step_num)
                except (ValueError, IndexError):
                    continue
        step_list.sort()
        print(f"自动检测到 {len(step_list)} 个step: {step_list}")
    
    if not step_list:
        raise ValueError(f'No evaluation steps found in {model_eval_dir}')
    
    # 遍历所有step
    uploaded_count = 0
    for step in step_list:
        step_dir = os.path.join(model_eval_dir, f'step_{step}')
        
        if not os.path.exists(step_dir):
            print(f'警告: Step {step} 的结果目录不存在: {step_dir}')
            continue
        
        greedy_data_path = os.path.join(step_dir, 'greedy_data_Overall_results.jsonl')
        temp_data_path = os.path.join(step_dir, 'temp_data_Overall_results.jsonl')
        
        # 检查文件是否存在
        if not os.path.exists(greedy_data_path):
            print(f'警告: Step {step} 的greedy_data结果文件不存在: {greedy_data_path}')
            continue
        
        if not os.path.exists(temp_data_path):
            print(f'警告: Step {step} 的temp_data结果文件不存在: {temp_data_path}')
            continue
        
        # 读取评测结果
        eval_data = {}
        try:
            d1, avg_d1 = read_eval(greedy_data_path, greedy=True)
            d2, avg_d2 = read_eval(temp_data_path, greedy=False)
            
            # 计算总体平均分
            avg = avg_d1 + avg_d2
            avg_res = sum([float(x) for x in avg]) / len(avg) if len(avg) > 0 else 0.0
            
            eval_data.update(d1)
            eval_data.update(d2)
            eval_data.update({"AVG": round(avg_res, 2)})
            
            # 上传到tensorboard，使用step作为x轴
            logger.log(data=eval_data, step=step)
            uploaded_count += 1
            print(f'✓ Step {step}: AVG={avg_res:.2f}%')
            
        except Exception as e:
            print(f'错误: 处理 Step {step} 时出错: {e}')
            continue
    
    print(f'\n成功上传 {uploaded_count}/{len(step_list)} 个step的评测结果到tensorboard')
    print(f'Tensorboard日志路径: {tb_path}')
    print(f'查看结果: tensorboard --logdir={tb_path}')


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='按step上传评测结果到tensorboard')
    parser.add_argument(
        "--model_name",
        type=str,
        default="filter_data_Qwen3-4B-Base",
        help="模型名称，例如 'filter_data_Qwen3-4B-Base'"
    )
    parser.add_argument(
        "--step_list",
        type=int,
        nargs='+',
        default=None,
        help="要上传的step列表，例如 --step_list 5 10 15 20。如果不指定，则自动检测所有可用的step"
    )
    args = parser.parse_args()
    tb_step(args.model_name, args.step_list)