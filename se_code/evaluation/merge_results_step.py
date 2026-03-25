import pandas as pd
import os


eval_dir='/root/users/ycy/saved_results/evaluation'
model_name='RR-Zero-V1-Step15-DeepScaleR'
eval_dir=os.path.join(eval_dir, model_name)
model_step_list=os.listdir(eval_dir)
result_list=['greedy_data_Overall_results.jsonl', 'temp_data_Overall_results.jsonl']
res = []

# Collect all data from all models
for model_step in model_step_list:
    model_path=os.path.join(eval_dir, model_step)
    if not os.path.isdir(model_path):
        continue
    
    for result_name in result_list:
        result_path=os.path.join(model_path, result_name)
        if not os.path.exists(result_path):
            continue
            
        if result_name == 'temp_data_Overall_results.jsonl':
            saved_item = ['model', 'data_source', 'checked_mean@32']
            result = pd.read_json(result_path, lines=True)
            result = result[saved_item]
            result['step'] = [model_step] * len(result)
            res.append(result)
            
        elif result_name == 'greedy_data_Overall_results.jsonl':
            saved_item = ['model', 'data_source', 'checked_mean@1']
            result = pd.read_json(result_path, lines=True)
            result = result[saved_item]
            result['step'] = [model_step] * len(result)
            res.append(result)

# Process results separately and merge them
if res:
    df_all = pd.concat(res, ignore_index=True)
    
    # Separate data by metric type
    df_mean1 = df_all[['model', 'data_source', 'checked_mean@1','step']].dropna(subset=['checked_mean@1']) if 'checked_mean@1' in df_all.columns else pd.DataFrame()
    df_mean32 = df_all[['model', 'data_source', 'checked_mean@32','step']].dropna(subset=['checked_mean@32']) if 'checked_mean@32' in df_all.columns else pd.DataFrame()
    
    # Create pivot tables for each metric
    df_pivoted_1 = None
    df_pivoted_32 = None
    
    if not df_mean1.empty:
        df_pivoted_1 = df_mean1.pivot_table(
            index='step',
            columns='data_source',
            values='checked_mean@1',
            aggfunc='first'
        ).reset_index()
        # Rename columns to distinguish metrics
        new_cols = {col: f"{col}_mean@1" for col in df_pivoted_1.columns if col != 'step'}
        df_pivoted_1 = df_pivoted_1.rename(columns=new_cols)
    
    if not df_mean32.empty:
        df_pivoted_32 = df_mean32.pivot_table(
            index='step',
            columns='data_source',
            values='checked_mean@32',
            aggfunc='first'
        ).reset_index()
        # Rename columns to distinguish metrics
        new_cols = {col: f"{col}_mean@32" for col in df_pivoted_32.columns if col != 'step'}
        df_pivoted_32 = df_pivoted_32.rename(columns=new_cols)
    
    # Merge both results
    if df_pivoted_1 is not None and df_pivoted_32 is not None:
        df_merged = pd.merge(df_pivoted_1, df_pivoted_32, on='step', how='outer')
    elif df_pivoted_1 is not None:
        df_merged = df_pivoted_1
    elif df_pivoted_32 is not None:
        df_merged = df_pivoted_32
    else:
        df_merged = None
    
    if df_merged is not None:
        # Add AVG column - calculate average of all numeric columns for each row
        numeric_cols = df_merged.select_dtypes(include=[float, int]).columns
        df_merged['AVG'] = df_merged[numeric_cols].mean(axis=1).round(2)
        df_merged['step_num'] = df_merged['step'].str.extract(r'(\d+)').astype(int)
        df_merged = df_merged.sort_values(by='step_num').drop(columns='step_num').reset_index(drop=True)
        # ✅ 指定 data_source 列顺序
            # ✅ 指定列顺序
        desired_order = [
            "MATH500_mean@1",
            "Minerva_mean@1",
            "OlympiadBench_mean@1",
            "AIME24_mean@32",
            "AIME25_mean@32",
            "AMC23_mean@1"
        ]

        # 提取 step 作为主索引
        cols_in_df = df_merged.columns.tolist()
        fixed_order = ["step"] + [col for col in desired_order if col in cols_in_df]

        # 其他列（未在 desired_order 中的）
        other_cols = [c for c in cols_in_df if c not in fixed_order and c != "AVG"]

        # 重新组织列顺序
        df_merged = df_merged[fixed_order + other_cols + ["AVG"]]
    
        # Save merged results to CSV and Excel
        csv_file = os.path.join(eval_dir, f'{model_name}_evolove_eval_results.csv')
        excel_file = os.path.join(eval_dir, f'{model_name}_evolove_eval_results.xlsx')
        
        df_merged.to_csv(csv_file, index=False)
        df_merged.to_excel(excel_file, index=False)
        
        print(f"Results saved to: {csv_file}")
        print(f"Results saved to: {excel_file}")
        print("\nResults preview:")
        print(df_merged)
    else:
        print("No results found!")
else:
    print("No results found!")
