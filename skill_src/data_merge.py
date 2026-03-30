import os
import json
import pandas as pd
import random
def data_merge(args):
    data_path_dir = args.data_path_dir
    save_path_dir = args.save_path_dir
    data_list = os.listdir(data_path_dir)
    datas = []
    for data_file in data_list:
        if data_file.endswith(".json"):
            with open(os.path.join(data_path_dir, data_file), "r") as f:
                datas.extend(json.load(f))
    os.makedirs(save_path_dir, exist_ok=True)
    #Please reason step by step, and put your final answer within \\boxed{}.
    instruction = "Please reason step by step, and put your final answer within \\boxed{}."
    # if args.hybrid_data:
    #     data_path='prompt_weakness/weakness_data_pool.json'
    #     with open(data_path,'r') as f:
    #         hybrid_data_dict = json.load(f)
    #         hybrid_data = [
    #             row
    #             for row in hybrid_data_dict.values()
    #         ]
    # 分开收集真实数据和合成数据
    real_data = []
    synthetic_data = []
    for idx, item in enumerate(datas):
        if item["score"] == 0:
            if item["is_synthetic"]:
                synthetic_data.append({
                    'data_source': f'Challenger_{args.exp_name}',
                    'problem':item["question"],
                    'prompt':[
                        {'role': 'system',   'content': instruction},
                        {
                        "role": "user",
                        # 'content':item["question"] + ' ' + instruction
                        'content':item["question"]
                    }],
                    'reward_model':{
                        'style':'rule', 
                    },
                    'ability': 'math',
                    'extra_info':{
                        'idx':idx,
                        'raw_data_source':item["data_source"],
                        'reference_question':item["reference_question"],
                        'question':item["question"],
                        'answer':item["answer"],
                        'score':item["score"],
                        'gen_q_prompt':item['prompt']
                    }
                })
            
            if not item["is_synthetic"] and args.hybrid_data:
                example=item["example"]
                real_data.append(example)

    # 真实数据全部保留，按 problem 字段记录
    seen_problems = set()
    for item in real_data:
        problem = item.get('problem')
        seen_problems.add(problem)
    #print(f'真实数据: {len(real_data)}')
    # 合成数据去重：排除与真实数据重复的 problem
    synthetic_cnt_before = len(synthetic_data)
    unique_synthetic_data = []
    for item in synthetic_data:
        problem = item.get('problem')
        if problem not in seen_problems:
            seen_problems.add(problem)
            unique_synthetic_data.append(item)
    
    print(f'真实数据: {len(real_data)}, 合成数据去重前: {synthetic_cnt_before}, 去重后: {len(unique_synthetic_data)}')
    
    # 保证真实数据出现在前 2 * len(real_data) 的范围内
    # 取 1 * len(real_data) 个合成数据与真实数据混合
    front_synthetic_cnt = min(len(real_data), len(unique_synthetic_data))
    front_synthetic = unique_synthetic_data[:front_synthetic_cnt]
    remain_synthetic = unique_synthetic_data[front_synthetic_cnt:]
    
    # 前半部分：真实数据 + 部分合成数据，shuffle 后放前面
    front_data = real_data + front_synthetic
    random.shuffle(front_data)
    
    # 后半部分：剩余合成数据，shuffle
    random.shuffle(remain_synthetic)
    
    # 拼接
    final_data = front_data + remain_synthetic
    print(f'前段数据(含全部真实): {len(front_data)}, 后段合成数据: {len(remain_synthetic)}, 总数: {len(final_data)}')        
    with open(f'{save_path_dir}/train_data.jsonl', 'w',encoding='utf-8') as f:
        for line in final_data:
            f.write(json.dumps(line,ensure_ascii=False) + "\n")
    # saved to parquet
    df = pd.DataFrame(final_data)
    df.to_parquet(f'{save_path_dir}/train_data.parquet')


            
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path_dir", type=str, default="/home/ycy/data1/saved_results/Challenger/qr_Rule_gqweakness_Qwen3-4B-Base-V1/gen_data", help="")
    parser.add_argument("--save_path_dir", type=str, default="/home/ycy/data1/Self-evolving-Agent/se_code_auto/prompt_weakness", help="")
    parser.add_argument("--exp_name", type=str, default="test", help="")
    parser.add_argument("--hybrid_data", action="store_true", help="")
    #parser.add_argument("--real_data_ratio", type=float, default=1.0, help="")
    args = parser.parse_args()
    #print(f"hybrid_data: {args.hybrid_data}")

    data_merge(args)
    
    