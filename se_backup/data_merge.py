import os
import json
import pandas as pd
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
    final_data=[]
    for idx, item in enumerate(datas):
        if item["score"] == 0:
            final_data.append({
            'data_source': f'Challenger_{args.exp_name}',
            'topic':item['topic'],
            'prompt':[
                {
                    'role': 'system',
                    'content':instruction
                },
                {
                "role": "user",
                'content':item["question"] 
            }],
            'reward_model':{
                'style':'rule', 
            },
            'ability': 'math',
            'extra_info':{
                'idx':idx,
                'question':item["question"],
                'answer':item["answer"],
                'score':item["score"]
            }
        })

    
    with open(f'{save_path_dir}/train_data.jsonl', 'w',encoding='utf-8') as f:
        for line in final_data:
            f.write(json.dumps(line,ensure_ascii=False) + "\n")
    # saved to parquet
    df = pd.DataFrame(final_data)
    df.to_parquet(f'{save_path_dir}/train_data.parquet')


            
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path_dir", type=str, default="", help="")
    parser.add_argument("--save_path_dir", type=str, default="", help="")
    parser.add_argument("--exp_name", type=str, default="", help="")
    args = parser.parse_args()
    data_merge(args)
    
    