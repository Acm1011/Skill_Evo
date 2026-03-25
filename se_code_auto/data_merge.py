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
    final_data=[]
    for idx, item in enumerate(datas):
        if item["score"] == 0:
            final_data.append({
            'data_source': f'Challenger_{args.exp_name}',
            'prompt':[
                {
                "role": "user",
                'content':item["question"] + ' ' + instruction
            }],
            'reward_model':{
                'style':'rule', 
            },
            'ability': 'math',
            'extra_info':{
                'idx':idx,
                'reference_question':item["reference_question"],
                'question':item["question"],
                'answer':item["answer"],
                'score':item["score"],
                'gen_q_prompt':item['prompt']
            }
        })

    # if args.hybrid_data:
    #     all_cnt = 3*128*20
    #     sample_cnt = 1*128*20
    #     ratio=0.1
    #     hybrid_data = random.sample(hybrid_data, int(all_cnt*ratio))
    #     print(f'hybrid real data cnt: {len(hybrid_data)}')
    #     sample_data = final_data[:sample_cnt]
    #     remain_data = final_data[sample_cnt:]
    #     train_data = random.shuffle(hybrid_data + sample_data)
    #     final_data = train_data + remain_data
    #     print(f'{len(sample_data)=}, {len(remain_data)=} {len(hybrid_data)=}, {len(train_data)=}, {len(final_data)=}')
        
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
    args = parser.parse_args()
    data_merge(args)
    
    