import pandas as pd
import os


eval_dir='/root/users/ycy/saved_results/evaluation'
model_list=os.listdir(eval_dir)
result_list=['greedy_data_Overall_results.jsonl', 'temp_data_Overall_results.jsonl']
res = []
st='prompt2_se'
# Collect all data from all models
for model_name in model_list:
    if not (model_name.startswith(st) or model_name.startswith('Qwen3')):
        continue
    model_path=os.path.join(eval_dir, model_name)
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
            res.append(result)
        elif result_name == 'greedy_data_Overall_results.jsonl':
            saved_item = ['model', 'data_source', 'checked_mean@1']
            result = pd.read_json(result_path, lines=True)
            result = result[saved_item]
            res.append(result)

# Process results separately and merge them
if res:
    df_all = pd.concat(res, ignore_index=True)
    
    # Separate data by metric type
    df_mean1 = df_all[['model', 'data_source', 'checked_mean@1']].dropna(subset=['checked_mean@1']) if 'checked_mean@1' in df_all.columns else pd.DataFrame()
    df_mean32 = df_all[['model', 'data_source', 'checked_mean@32']].dropna(subset=['checked_mean@32']) if 'checked_mean@32' in df_all.columns else pd.DataFrame()
    
    # Create pivot tables for each metric
    df_pivoted_1 = None
    df_pivoted_32 = None
    
    if not df_mean1.empty:
        df_pivoted_1 = df_mean1.pivot_table(
            index='model',
            columns='data_source',
            values='checked_mean@1',
            aggfunc='first'
        ).reset_index()
        # Rename columns to distinguish metrics
        new_cols = {col: f"{col}_mean@1" for col in df_pivoted_1.columns if col != 'model'}
        df_pivoted_1 = df_pivoted_1.rename(columns=new_cols)
    
    if not df_mean32.empty:
        df_pivoted_32 = df_mean32.pivot_table(
            index='model',
            columns='data_source',
            values='checked_mean@32',
            aggfunc='first'
        ).reset_index()
        # Rename columns to distinguish metrics
        new_cols = {col: f"{col}_mean@32" for col in df_pivoted_32.columns if col != 'model'}
        df_pivoted_32 = df_pivoted_32.rename(columns=new_cols)
    
    # Merge both results
    if df_pivoted_1 is not None and df_pivoted_32 is not None:
        df_merged = pd.merge(df_pivoted_1, df_pivoted_32, on='model', how='outer')
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
        # 指定横向列顺序
        desired_order = [
            "AMC23_mean@1",
            "MATH500_mean@1",
            "Minerva_mean@1",
            "OlympiadBench_mean@1",
            "AIME24_mean@32",
            "AIME25_mean@32"
        ]

        # 按列顺序重排
        cols_in_df = df_merged.columns.tolist()
        ordered_cols = ["model"] + [col for col in desired_order if col in cols_in_df] + ["AVG"]
        remaining_cols = [col for col in cols_in_df if col not in ordered_cols]
        df_merged = df_merged[[c for c in ordered_cols if c in df_merged.columns] + remaining_cols]

        # Save merged results to CSV and Excel
        csv_file = os.path.join(eval_dir, f'{st}_evolove_eval_results.csv')
        excel_file = os.path.join(eval_dir, f'{st}_evolove_eval_results.xlsx')
        
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
