from verl.utils.tracking import Tracking
import os
import pandas as pd


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
    

def tb(prefix, step, temperature):
    tb_path_dir='/root/users/ycy/saved_results/eval_tb_log'
    os.makedirs(tb_path_dir,exist_ok=True)
    eval_results_dir='/root/users/ycy/saved_results/evaluation'
    n_iter=0
    for path in os.listdir(eval_results_dir):
        if path.startswith(prefix):
            n_iter += 1
    tb_path=os.path.join(tb_path_dir, f'{prefix}-step_{step}-temperature_{temperature}')
    os.environ['TENSORBOARD_DIR']=tb_path
    logger=Tracking(project_name='Se', experiment_name=prefix, default_backend=['console','tensorboard'])

    for i in range(0, n_iter+1):
        if i == 0:
            suff='Qwen3-4B-Base'
        else:
            suff=f'{prefix}-V{i}'
            
        eval_path=os.path.join(eval_results_dir, suff)
        if not os.path.exists(eval_path):
            raise ValueError(f'{eval_path=} not exists')
        eval_data = {}
        greedy_data_path=os.path.join(eval_path, 'greedy_data_Overall_results.jsonl')
        tmp_data_path=os.path.join(eval_path, 'temp_data_Overall_results.jsonl')
        d1, avg_d1=read_eval(greedy_data_path, greedy=True)
        d2, avg_d2=read_eval(tmp_data_path, greedy=False)
        avg = avg_d1 + avg_d2
        avg_res = sum(avg) / len(avg) if len(avg) > 0 else 0.0
        eval_data.update(d1)
        eval_data.update(d2)
        eval_data.update({"AVG":round(avg_res, 2)})
        logger.log(data=eval_data,step=i)


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
    args = parser.parse_args()
    tb(args.prefix, args.step, args.temperature)