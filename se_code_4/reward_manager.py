from collections import defaultdict,Counter

import torch
import re
from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
from typing import Dict, List,Optional, Any
import json
from mathruler.grader import extract_boxed_content, grade_answer
import os
import time
import random
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.cluster import AgglomerativeClustering
import numpy as np



def custom_extract_boxed_content(text: str) -> str:
    """
    Extracts answers in \\boxed{}.
    """
    depth = 0
    start_pos = text.rfind(r"\boxed{")
    end_pos = -1
    if start_pos != -1:
        content = text[start_pos + len(r"\boxed{") :]
        for i, char in enumerate(content):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1

            if depth == -1:  # exit
                end_pos = i
                break

    if end_pos != -1:
        return content[:end_pos].strip()

    return None


def _bleu_distance_matrix(sentences):
    n = len(sentences)
    dist = np.zeros((n, n))
    smoother = SmoothingFunction().method1
    for i in range(n):
        for j in range(i, n):
            if i == j:
                score = 1.0
            else:
                ref = [sentences[j].split()]
                hyp = sentences[i].split()
                score = sentence_bleu(ref, hyp, smoothing_function=smoother)
                # sentence_bleu may return float or list[float], ensure we get a float
                score = score[0] if isinstance(score, list) else score
            dist[i, j] = dist[j, i] = 1 - score
    return dist

def cluster_share_per_problem(
        problems,
        distance_threshold: float = 0.5,
        linkage: str = "average"):
    if not problems:
        return []
    print('start clustering')
    start_time = time.time()
    dist_mat = _bleu_distance_matrix(problems)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage=linkage
    )
    labels = clustering.fit_predict(dist_mat)
    print(f'end clustering, time: {time.time() - start_time}')
    total = len(problems)
    cluster_size = Counter(labels)
    cluster_ratio = {lab: sz / total for lab, sz in cluster_size.items()}

    proportions = [cluster_ratio[lab] for lab in labels]
    return proportions

def generate_temp_filename(storage_path:str, prefix="temp", suffix=".json"):
    timestamp = int(time.time() * 1000) 
    rand_part = random.randint(0, 99999)
    os.makedirs(f"{storage_path}/temp_results", exist_ok=True)
    return f"{storage_path}/temp_results/{prefix}_{timestamp}_{rand_part}{suffix}"

def split_list(lst, n=2):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

os.environ["NO_PROXY"] = "0.0.0.0,127.0.0.1"

def fetch(index,i, question_reward):
    response = requests.get(f"http://0.0.0.0:{5000+index}/hello?name={i}&question_reward={question_reward}")
    print(response)
    return True

def generate_results(data, storage_path: str, question_reward: str):
    # 将数据分成4份
    datas = split_list(data, 2)
    random_names = [generate_temp_filename(storage_path=storage_path, prefix=f"temp_{i}") for i in range(2)]
    
    # 保存数据到临时文件
    for i in range(2):
        with open(random_names[i], 'w') as f:
            json.dump(datas[i], f, indent=4)

    final_results = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(fetch, i, random_names[i], question_reward) for i in range(2)]

        for future in as_completed(futures):
            print(future.result())

    for i in range(2):
        with open(random_names[i].replace('.json','_results.json'),'r') as f:
            final_results.extend(json.load(f))
    for i in range(2):
        os.remove(random_names[i].replace('.json','_results.json'))
    return final_results


@register("challenger")
class ChallengerRewardManager(AbstractRewardManager):
    """The reward manager."""
    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key='challenger',
        storage_path:str="",
        question_reward:str="R_Zero",
        group_question_repetion_penalty=True
    ) -> None:
        assert storage_path is not None, "storage_path must be provided"
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.storage_path = storage_path
        self.question_reward=question_reward
        self.group_question_repetion_penalty=group_question_repetion_penalty
        os.makedirs(self.storage_path, exist_ok=True)
    def compute_score(self, predicts, storage_path:str, step=0):
        results = []
        for i in range(len(predicts)):
            questions = re.findall(r"<question>(.*?)</question>", predicts[i], re.DOTALL)
            answers = re.findall(r"<answer>(.*?)</answer>", predicts[i], re.DOTALL)
            if questions and answers:
                try:
                    question = questions[-1].strip()
                    answer = answers[-1].strip()
                    results.append({"idx":i, "question": question, "answer": answer})
                except:
                    results.append({"idx":i, "question": "", "answer": ""})
            else:
                results.append({"idx":i, "question": "", "answer": ""})

        final_results = generate_results(results, storage_path=storage_path, question_reward=self.question_reward)
        if self.group_question_repetion_penalty:
            penalty = cluster_share_per_problem([result['question'] for result in final_results], distance_threshold=0.5)
        else:
            penalty = [0] * len(final_results)
        # print(penalty)
        assert len(penalty) == len(final_results), f'{len(penalty)=}\n {len(final_results)=}'
        scores = []
        saved_results = []
        for i in range(len(final_results)):
            # Use uncertrainity_reward from vLLM server response, fallback to 0 if not available
            base_score = final_results[i].get("reward", 0)
            final_score = max(0, base_score - penalty[i]) if final_results[i]['question'] else 0
            scores.append({"score": final_score, "format": 1 if final_results[i]['question'] else 0,"repetition_penalty": penalty[i]})
            saved_results.append(
                {
                    "idx": i,
                    'step': step,
                    'Challenger_rollout': predicts[i],
                    'extracted_question':final_results[i]['question'],
                    'reward_info': final_results[i]['reward_info'],
                    'reward': final_results[i]['reward'],
                    'final_reward':final_score,
                    'error':final_results[i].get('error', None),
                }
            )
        reward_info_path_dir = f"{self.storage_path}/reward_info/"
        os.makedirs(reward_info_path_dir, exist_ok=True)
        step_str = str(step).zfill(3)
        with open(f"{reward_info_path_dir}/expdata_step_{step_str}.jsonl", 'w', encoding='utf-8') as f:
            for result in saved_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        return scores
    
    def __call__(self, data: DataProto, return_dict: bool = False, step: int = 0):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        qeurys = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            #valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            #valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            #prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]
            qeurys.append(response_str)
            
        results = self.compute_score(qeurys, self.storage_path, step)
        for i in range(len(results)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            

            score: float
            result = results[i]
            
            if isinstance(result, dict):
                score = result["score"]
                
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result
                reward_extra_info["acc"].append(score)

            reward = score
            # TODO: add reward post-processing
            reward_tensor[i, valid_response_length - 1] = reward
            
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

@register("challenger_gan")
class ChallengerGANRewardManager(AbstractRewardManager):
    """The reward manager."""
    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key='challenger',
        storage_path:str="",
    ) -> None:
        assert storage_path is not None, "storage_path must be provided"
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.storage_path = storage_path
        with open("/root/users/ycy/data/DeepMath-103K_t2q_s128.json", "r") as f:
            self.real_qs = json.load(f)
        os.makedirs(self.storage_path, exist_ok=True)
    def compute_score(self, predicts, storage_path:str, topics, step=0):
        results = []
        real_question = self.real_qs[topics[i]][random.randint(0, len(self.real_qs[topics[i]]) - 1)].strip()
        for i in range(len(predicts)):
            questions = re.findall(r"<question>(.*?)</question>", predicts[i], re.DOTALL)
            
            if questions:
                try:
                    question = questions[-1].strip()
                    results.append({"idx":i, "question": question,'real_question': real_question})
                except:
                    results.append({"idx":i, "question": "", "real_question": real_question})
            else:
                results.append({"idx":i, "question": "", "real_question": real_question})

        final_results = generate_results(results, storage_path=storage_path)
        penalty = cluster_share_per_problem([result['question'] for result in final_results], distance_threshold=0.5)
        # print(penalty)
        assert len(penalty) == len(final_results), f'{len(penalty)=}\n {len(final_results)=}'
        scores = []
        saved_results = []
        for i in range(len(final_results)):
            # Use uncertrainity_reward from vLLM server response, fallback to 0 if not available
            base_score = final_results[i].get("reward", 0)
            final_score = max(0, base_score - penalty[i]) if final_results[i]['question'] else 0
            scores.append({"score": final_score, "topic":topics[i], "format": 1 if final_results[i]['question'] else 0,"repetition_penalty": penalty[i]})
            saved_results.append(
                {
                    "idx": i,
                    'step': step,
                    "topic":topics[i],
                    'rollout_questions': predicts[i],
                    "extracted_question":final_results[i]['question'],
                    'sample_responses': final_results[i]['responses'],
                    'extracted_results': final_results[i]['extracted_results'],
                    'all_rewards': final_results[i]['rewards'],
                    'reward': final_results[i]['reward'],
                    'error':final_results[i].get('error', None),
                }
            )
        reward_info_path_dir = f"{self.storage_path}/reward_info/"
        os.makedirs(reward_info_path_dir, exist_ok=True)
        step_str = str(step).zfill(3)
        with open(f"{reward_info_path_dir}/expdata_step_{step_str}.jsonl", 'w', encoding='utf-8') as f:
            for result in saved_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        return scores
    
    def __call__(self, data: DataProto, return_dict: bool = False, step: int = 0):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        qeurys = []
        topics = []
        target_levels=[]
        topics2infos=defaultdict(list)
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            topic = data_item.non_tensor_batch["topic"]
            target_level=data_item.non_tensor_batch["target_level"]

            prompt_length = prompt_ids.shape[-1]

            #valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            #valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            #prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]
            qeurys.append(response_str)
            topics.append(topic)
            target_levels.append(str(target_level))
            
        results = self.compute_score(qeurys, self.storage_path, topics, step)
        for i in range(len(results)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            

            score: float
            result = results[i]
            
            if isinstance(result, dict):
                score = result["score"]
                
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result
                reward_extra_info["acc"].append(score)

            reward = score
            # TODO: add reward post-processing
            reward_tensor[i, valid_response_length - 1] = reward
            assert result['topic'] == topics[i], f"{result['topic']=}, {topics[i]=}"
            topics2infos[result['topic']].append(reward)
        topics_infos = {
                        topic:{
                            'reward':float(np.mean(reward)),
                            'count':len(reward),
                        } 
                        for topic, reward in topics2infos.items()
                    }
        print(f'topics_infos')
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
                'topics_infos':topics_infos
            }
        else:
            return reward_tensor



@register("challenger_entropy")
class ChallengerEntropyRewardManager(AbstractRewardManager):
    """The reward manager."""
    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key='challenger',
        storage_path:str="",
    ) -> None:
        assert storage_path is not None, "storage_path must be provided"
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)
    def compute_score(self, predicts, storage_path:str, topics, step=0):
        results = []
        for i in range(len(predicts)):
            questions = re.findall(r"<question>(.*?)</question>", predicts[i], re.DOTALL)
            if questions:
                try:
                    question = questions[-1].strip()
                    results.append({"idx":i, "question": question})
                except:
                    results.append({"idx":i, "question": "",})
            else:
                results.append({"idx":i, "question": ""})

        final_results = generate_results(results, storage_path=storage_path)
        penalty = cluster_share_per_problem([result['question'] for result in final_results], distance_threshold=0.5)
        # print(penalty)
        assert len(penalty) == len(final_results), f'{len(penalty)=}\n {len(final_results)=}'
        scores = []
        saved_results = []
        for i in range(len(final_results)):
            # Use uncertrainity_reward from vLLM server response, fallback to 0 if not available
            base_score = final_results[i].get("reward", 0)
            final_score = max(0,  base_score - penalty[i]) if final_results[i]['question'] else 0
            scores.append({
                    "score": final_score, 
                    "topic":topics[i], 
                    "format": 1 if final_results[i]['question'] else 0,
                    'penalty_score':penalty[i]
                    }
                )
            saved_results.append(
                {
                    "idx": i,
                    'step': step,
                    "topic":topics[i],
                    'rollout_rsp': predicts[i],
                    'extracted_question':final_results[i]['question'],
                    'reward_info':final_results[i]['reward_info'],
                    'answer_counts':final_results[i]['answer_counts'],
                    'all_labels':final_results[i]['all_labels'],
                    'valid_labels':final_results[i]['valid_labels'],
                    'valid_labels_cnt':final_results[i]['valid_labels_cnt'],
                    'all_labels_cnt':final_results[i]['all_labels_cnt'],
                    'gen_reward':base_score,
                    'penalty_score':penalty[i],
                    'final_reward':final_score,
                    'majority_accuracy':final_results[i]['majority_accuracy'],
                    'uncertainty_reward':final_results[i]['uncertainty_reward'],
                    'R-Zero_reward':max(0,final_results[i]['uncertainty_reward'] - penalty[i]),
                    'error':final_results[i].get('error', None),
                }
            )
        reward_info_path_dir = f"{self.storage_path}/reward_info/"
        os.makedirs(reward_info_path_dir, exist_ok=True)
        step_str = str(step).zfill(3)
        with open(f"{reward_info_path_dir}/expdata_step_{step_str}.jsonl", 'w', encoding='utf-8') as f:
            for result in saved_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        return scores
    
    def __call__(self, data: DataProto, return_dict: bool = False, step: int = 0):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        qeurys = []
        topics = []
        target_levels=[]
        topics2infos=defaultdict(list)
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            topic = data_item.non_tensor_batch["topic"]
            target_level=data_item.non_tensor_batch["target_level"]

            prompt_length = prompt_ids.shape[-1]

            #valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            #valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            #prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]
            qeurys.append(response_str)
            topics.append(topic)
            target_levels.append(str(target_level))
            
        results = self.compute_score(qeurys, self.storage_path, topics, step)
        for i in range(len(results)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            

            score: float
            result = results[i]
            
            if isinstance(result, dict):
                score = result["score"]
                
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result
                reward_extra_info["acc"].append(score)

            reward = score
            # TODO: add reward post-processing
            reward_tensor[i, valid_response_length - 1] = reward
            assert result['topic'] == topics[i], f"{result['topic']=}, {topics[i]=}"
            topics2infos[result['topic']].append(reward)
        topics_infos = {
                        topic:{
                            'reward':float(np.mean(reward)),
                            'count':len(reward),
                        } 
                        for topic, reward in topics2infos.items()
                    }
        #print(f'topics_infos')
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
                'topics_infos':topics_infos
            }
        else:
            return reward_tensor

@register("challenger_rule")
class ChallengerRuleRewardManager(AbstractRewardManager):
    """The reward manager."""
    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key='challenger',
        storage_path:str="",
    ) -> None:
        assert storage_path is not None, "storage_path must be provided"
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)
    def compute_score(self, predicts, storage_path:str, topics, step=0):
        results = []
        for i in range(len(predicts)):
            questions = re.findall(r"<question>(.*?)</question>", predicts[i], re.DOTALL)
            if questions:
                try:
                    question = questions[-1].strip()
                    results.append({"idx":i, "question": question})
                except:
                    results.append({"idx":i, "question": "",})
            else:
                results.append({"idx":i, "question": ""})

        final_results = generate_results(results, storage_path=storage_path)
        #penalty = cluster_share_per_problem([result['question'] for result in final_results], distance_threshold=0.5)
        # print(penalty)
        #assert len(penalty) == len(final_results), f'{len(penalty)=}\n {len(final_results)=}'
        scores = []
        saved_results = []
        for i in range(len(final_results)):
            # Use uncertrainity_reward from vLLM server response, fallback to 0 if not available
            base_score = final_results[i].get("reward", 0)
            #final_score = max(0,  base_score - penalty[i]) if final_results[i]['question'] else 0
            final_score = base_score if final_results[i]['question'] else 0
            scores.append({
                    "score": final_score, 
                    "topic":topics[i], 
                    "format": 1 if final_results[i]['question'] else 0,
                    #'penalty_score':penalty[i]
                    }
                )
            saved_results.append(
                {
                    "idx": i,
                    'step': step,
                    "topic":topics[i],
                    'rollout_rsp': predicts[i],
                    'extracted_question':final_results[i]['question'],
                    'reward_info':final_results[i]['reward_info'],
                    'reward':final_results[i]['reward'],
                    'error':final_results[i].get('error', None),
                }
            )
        reward_info_path_dir = f"{self.storage_path}/reward_info/"
        os.makedirs(reward_info_path_dir, exist_ok=True)
        step_str = str(step).zfill(3)
        with open(f"{reward_info_path_dir}/expdata_step_{step_str}.jsonl", 'w', encoding='utf-8') as f:
            for result in saved_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        return scores
    
    def __call__(self, data: DataProto, return_dict: bool = False, step: int = 0):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        qeurys = []
        topics = []
        target_levels=[]
        topics2infos=defaultdict(list)
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            topic = data_item.non_tensor_batch["topic"]
            target_level=data_item.non_tensor_batch["target_level"]

            prompt_length = prompt_ids.shape[-1]

            #valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            #valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            #prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]
            qeurys.append(response_str)
            topics.append(topic)
            target_levels.append(str(target_level))
            
        results = self.compute_score(qeurys, self.storage_path, topics, step)
        for i in range(len(results)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            

            score: float
            result = results[i]
            
            if isinstance(result, dict):
                score = result["score"]
                
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result
                reward_extra_info["acc"].append(score)

            reward = score
            # TODO: add reward post-processing
            reward_tensor[i, valid_response_length - 1] = reward
            assert result['topic'] == topics[i], f"{result['topic']=}, {topics[i]=}"
            topics2infos[result['topic']].append(reward)
        topics_infos = {
                        topic:{
                            'reward':float(np.mean(reward)),
                            'count':len(reward),
                        } 
                        for topic, reward in topics2infos.items()
                    }
        #print(f'topics_infos')
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
                'topics_infos':topics_infos
            }
        else:
            return reward_tensor



@register("solver_base")
class SolverBaseRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        storage_path:str="",
        filter_lower= 0.0,
        filter_high= 1.0,
        data_pool_path=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.storage_path = storage_path
        self.low = filter_lower
        self.high = filter_high
        self.data_pool_path = data_pool_path

        
    def compute_score(
        self,
        solution_str: str,
        ground_truth: str,
    ) -> dict[str, Any]:
        """Compute the reward score for a solution.

        Args:
            solution_str: The solution string
            ground_truth: The ground truth answer

        Returns:
            Reward score (1.0 for correct, -1.0 for incorrect)
        """
        # Limit solution length for efficiency
        solution_str = solution_str[-300:]  # The longest answer in MATH-500 has 159 characters

        # Verify the solution
        if not isinstance(ground_truth, list):
            ground_truth = [ground_truth]
        correct = False
        pred = custom_extract_boxed_content(solution_str)
        for gt in ground_truth:
            if pred is None:
                continue
            correct = grade_answer(str(pred), str(gt))
            if correct:
                break

        reward = 1.0 if correct  else 0.0
        acc = reward

        return {
            "score": reward,
            "acc": acc,
            "pred": pred if pred is not None else 'None',
        }
 
    def __call__(self, data: DataProto, return_dict: bool = False, step: int = 0):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]
        #topics = data.non_tensor_batch["topic"] if self.num_examine == 0 else data.non_tensor_batch["data_source"]
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        uids = data.non_tensor_batch["uid"]
        reward_extra_info = defaultdict(list)
        responses = []
        responses_length = []
        group_acc=defaultdict(list)
        group_data=defaultdict(list)

        reward_infos = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            group_data[uids[i]].append(data_item)
            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            responses_length.append(valid_response_length)
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]
            
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            result=self.compute_score(solution_str=response_str, ground_truth=ground_truth)
            responses.append(response_str)
            if random.randint(0,64)==0:
                print(rf"{i=},\n {response_str=},\n {ground_truth=},\n {result=}")
            score: float
            if isinstance(result, dict):
                score = result["score"]
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result
                reward_extra_info["acc"].append(score)
            reward = score
            reward_tensor[i, valid_response_length - 1] = reward
            group_acc[uids[i]].append(score)
            reward_infos.append(
                {
                    'idx':i,
                    "step": step,
                    "style": "val" if self.num_examine > 0 else "exp",
                    "question": prompt_str,
                    "response": response_str,
                    "pred": result.get("pred", ""),
                    "ground_truth": ground_truth,
                    "reward": reward,
                }
            )
        if self.data_pool_path:
            if os.path.exists(self.data_pool_path):
                with open(self.data_pool_path, 'r', encoding='utf-8') as f:
                    data_pool = json.load(f)
                print(f"{step=}, {self.data_pool_path} found, load data pool successfully")
            else:
                data_pool = {}
                print(f"{step=},{self.data_pool_path} not found, create new data pool")
            #acc_list = reward_extra_info["acc"]
            for uid, data_items in group_data.items():
                data_item = data_items[0]  # DataProtoItem
                idxs = [d.non_tensor_batch["extra_info"]["idx"] for d in data_items]
                assert len(set(idxs)) == 1, f"sum of idxs is not equal to idxs[0] * len(idxs): {sum(idxs)=}, {idxs[0]=}, {len(idxs)=}"
                acc_list=group_acc[uid]
                assert len(acc_list) == len(data_items), f"length of acc_list and data_items is not equal: {len(acc_list)=}, {len(data_items)=}"
                acc = sum(acc_list) / len(acc_list) if len(acc_list) > 0 else 0.0
                extra_info=data_item.non_tensor_batch["extra_info"]
                idx=extra_info["idx"]
                instruction = "Please reason step by step, and put your final answer within \\boxed{}."
                if idx in data_pool: # 已存在
                    old_acc = data_pool[idx]['extra_info']['acc']
                    if acc > old_acc:
                        data_pool.pop(idx)
                        if random.randint(0,100)==0:
                            print(f"{idx=}, current {acc=}, old {old_acc=}, pop data pool")
                if acc < 0.5:
                    data_source=data_item.non_tensor_batch["data_source"]
                    q = extra_info["problem"]
                    index=extra_info["idx"]
                    prompt = [{"role": "user", "content": q + " " + instruction}]
                    reward_model=data_item.non_tensor_batch['reward_model']
                    ability="math"
                    extra_info.update({"acc":acc})
                    d={
                        "data_source":data_source,
                        "prompt":prompt,
                        "reward_model":reward_model,
                        "ability":ability,
                        "extra_info":extra_info
                    }
                    data_pool[index]=d
                    if random.randint(0,100)==0:
                        print(f"{index=}, \\n {acc=}, \\n{d=},")
            with open(self.data_pool_path, 'w', encoding='utf-8') as f:
                json.dump(data_pool, f, ensure_ascii=False, indent=4)
        #reward_infos =[example.update({'uid2group_acc':uid2group_acc[uids[i]]}) for i, example in enumerate(reward_infos)]
        #/root/users/ycy/Self-evolving-Agent/saved_results/Solver/Qwen3-4B-Base-V1
        reward_extra_info['reward_infos'] = reward_infos
        reward_info_path_dir = f"{self.storage_path}/reward_info/"
        step_str = str(step).zfill(3)
        if self.num_examine > 0:
            os.makedirs(f"{reward_info_path_dir}/valdata", exist_ok=True)
            with open(f"{reward_info_path_dir}/valdata/step_{step_str}.jsonl", 'w', encoding='utf-8') as f:
                for reward_info in reward_infos:
                    f.write(json.dumps(reward_info, ensure_ascii=False) + '\n')
        else:
            os.makedirs(f"{reward_info_path_dir}/expdata", exist_ok=True)
            with open(f"{reward_info_path_dir}/expdata/step_{step_str}.jsonl", 'w', encoding='utf-8') as f:
                for reward_info in reward_infos:
                    f.write(json.dumps(reward_info, ensure_ascii=False) + '\n')
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor


@register("solver_rule")
class SolverRewardManager(AbstractRewardManager):
    """The reward manager."""
    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        storage_path:str="",
        filter_lower= 0.0,
        filter_high= 1.0,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.storage_path = storage_path
        self.low = filter_lower
        self.high = filter_high
        
    def compute_score(
        self,
        solution_str: str,
        ground_truth: str,
    ) -> dict[str, Any]:
        """Compute the reward score for a solution.

        Args:
            solution_str: The solution string
            ground_truth: The ground truth answer

        Returns:
            Reward score (1.0 for correct, -1.0 for incorrect)
        """
        # Limit solution length for efficiency
        solution_str = solution_str[-300:]  # The longest answer in MATH-500 has 159 characters

        # Verify the solution
        if not isinstance(ground_truth, list):
            ground_truth = [ground_truth]
        correct = False
        pred = custom_extract_boxed_content(solution_str)
        for gt in ground_truth:
            if pred is None:
                continue
            correct = grade_answer(str(pred), str(gt))
            if correct:
                break

        reward = 1.0 if correct  else 0.0
        acc = reward

        return {
            "score": reward,
            "acc": acc,
            "pred": pred if pred is not None else 'None',
        }
 
    def __call__(self, data: DataProto, return_dict: bool = False, step: int = 0):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        uids = data.non_tensor_batch["uid"]
        uid2labels = defaultdict(list)
        uid2all_labels = defaultdict(list)
        
        labels = []
        prompts = []
        responses = []
        responses_length = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            
            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            responses_length.append(valid_response_length)
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]
            label = custom_extract_boxed_content(response_str[-300:])
            uid2all_labels[uids[i]].append(label if label is not None else 'None') 

            if label is not None:
                uid2labels[uids[i]].append(label)
            
            prompts.append(prompt_str)
            responses.append(response_str)
        uid2ground_truths = defaultdict(lambda:None)   
        ground_truths = []
        for i in range(len(data)):
            if 'ground_truth' in data[i].non_tensor_batch['reward_model']:
                ground_truths.append(data[i].non_tensor_batch['reward_model']['ground_truth'])
            else:
                if uid2ground_truths[uids[i]] is not None:
                    ground_truths.append(uid2ground_truths[uids[i]])
                    continue
                answers_count = {}
                for res in list(uid2labels[uids[i]]):
                    if not res: continue
                    matched = False
                    for exist_ans in list(set(answers_count.keys())):
                        if res == exist_ans or ('no ' in res.lower() and 'no ' in exist_ans.lower()):
                            answers_count[exist_ans] += 1
                            matched = True
                            break
                        try:
                            is_match = False
                            is_match = grade_answer(str(res), str(exist_ans))
                            if is_match:
                                answers_count[exist_ans] += 1
                                matched = True
                                break
                        except Exception as e:
                            print(f"Error comparing '{res}' and '{exist_ans}': {e}")
                            continue
                    if not matched:
                        answers_count[res] = 1
                if not answers_count:
                    majority_ans, max_count = '', 0
                else:
                    majority_ans = max(answers_count, key=answers_count.get)
                    max_count = answers_count[majority_ans]
                uid2ground_truths[uids[i]]=majority_ans
                ground_truths.append(majority_ans)



               
        reward_infos = []
        uid2group_acc = defaultdict(list)
        for i in range(len(data)):
            if self.num_examine > 0: # val data
                expected_gt = data[i].non_tensor_batch['reward_model'].get('ground_truth', [])
                assert expected_gt == ground_truths[i], f"val data ground_truth is not correct: expected {expected_gt}, got {ground_truths[i]}" 
            
            result = self.compute_score(responses[i], ground_truths[i])
            score: float
            valid_response_length = responses_length[i]
            if isinstance(result, dict):
                score = result["score"]
                uid2group_acc[uids[i]].append(result["acc"])
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result
                uid2group_acc[uids[i]].append(score)
                reward_extra_info["acc"].append(score)
            reward = score
            reward_tensor[i, valid_response_length - 1] = reward
            
            reward_infos.append(
                {
                    'idx':i,
                    "step": step,
                    "style": "val" if self.num_examine > 0 else "exp",
                    "question": prompts[i],
                    'raw_question': data[i].non_tensor_batch['raw_prompt'],
                    "response": responses[i],
                    "pred": result.get("pred", ""),
                    "all_labels": uid2all_labels[uids[i]],
                    'filtered_labels': uid2labels[uids[i]],
                    "ground_truth(majority)": ground_truths[i],
                    "reward": reward,
                }
            )
            
        for i, example in enumerate(reward_infos):
            acc_mean=float(np.mean(uid2group_acc[uids[i]]))
            example.update({
                'uid2group_acc':uid2group_acc[uids[i]],
                'uid2acc_mean': acc_mean,
                'is_kept': bool(acc_mean >= self.low and acc_mean <= self.high)
            })
            #print(f'{i=},{example=}')
        

        reward_extra_info['reward_infos'] = reward_infos
        #reward_infos =[example.update({'uid2group_acc':uid2group_acc[uids[i]]}) for i, example in enumerate(reward_infos)]
        #/root/users/ycy/Self-evolving-Agent/saved_results/Solver/Qwen3-4B-Base-V1
        reward_info_path_dir = f"{self.storage_path}/reward_info/"
        step_str = str(step).zfill(3)
        if self.num_examine > 0:
            os.makedirs(f"{reward_info_path_dir}/valdata", exist_ok=True)
            with open(f"{reward_info_path_dir}/valdata/step_{step_str}.jsonl", 'w', encoding='utf-8') as f:
                for reward_info in reward_infos:
                    f.write(json.dumps(reward_info, ensure_ascii=False) + '\n')
        else:
            os.makedirs(f"{reward_info_path_dir}/expdata", exist_ok=True)
            with open(f"{reward_info_path_dir}/expdata/step_{step_str}.jsonl", 'w', encoding='utf-8') as f:
                for reward_info in reward_infos:
                    f.write(json.dumps(reward_info, ensure_ascii=False) + '\n')
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor


@register("solver")
class SolverRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        storage_path:str="",
        filter_lower= 0.0,
        filter_high= 1.0,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.storage_path = storage_path
        self.low = filter_lower
        self.high = filter_high
        
    def compute_score(
        self,
        solution_str: str,
        ground_truth: str,
    ) -> dict[str, Any]:
        """Compute the reward score for a solution.

        Args:
            solution_str: The solution string
            ground_truth: The ground truth answer

        Returns:
            Reward score (1.0 for correct, -1.0 for incorrect)
        """
        # Limit solution length for efficiency
        solution_str = solution_str[-300:]  # The longest answer in MATH-500 has 159 characters

        # Verify the solution
        if not isinstance(ground_truth, list):
            ground_truth = [ground_truth]
        correct = False
        pred = custom_extract_boxed_content(solution_str)
        for gt in ground_truth:
            if pred is None:
                continue
            correct = grade_answer(str(pred), str(gt))
            if correct:
                break

        reward = 1.0 if correct  else 0.0
        acc = reward

        return {
            "score": reward,
            "acc": acc,
            "pred": pred if pred is not None else 'None',
        }
 
    def __call__(self, data: DataProto, return_dict: bool = False, step: int = 0):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]
        #topics = data.non_tensor_batch["topic"] if self.num_examine == 0 else data.non_tensor_batch["data_source"]
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        uids = data.non_tensor_batch["uid"]
        uid2labels = defaultdict(list)
        uid2all_labels = defaultdict(list)
        
        labels = []
        prompts = []
        responses = []
        responses_length = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            
            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            responses_length.append(valid_response_length)
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]
            label = custom_extract_boxed_content(response_str[-300:])
            uid2all_labels[uids[i]].append(label if label is not None else 'None') 

            if label is not None:
                uid2labels[uids[i]].append(label)
            
            prompts.append(prompt_str)
            responses.append(response_str)
        uid2ground_truths = defaultdict(lambda:None)   
        ground_truths = []
        for i in range(len(data)):
            if 'ground_truth' in data[i].non_tensor_batch['reward_model']:
                ground_truths.append(data[i].non_tensor_batch['reward_model']['ground_truth'])
            else:
                if uid2ground_truths[uids[i]] is not None:
                    ground_truths.append(uid2ground_truths[uids[i]])
                    continue
                answers_count = {}
                for res in list(uid2labels[uids[i]]):
                    if not res: continue
                    matched = False
                    for exist_ans in list(set(answers_count.keys())):
                        if res == exist_ans or ('no ' in res.lower() and 'no ' in exist_ans.lower()):
                            answers_count[exist_ans] += 1
                            matched = True
                            break
                        try:
                            is_match = False
                            is_match = grade_answer(str(res), str(exist_ans))
                            if is_match:
                                answers_count[exist_ans] += 1
                                matched = True
                                break
                        except Exception as e:
                            print(f"Error comparing '{res}' and '{exist_ans}': {e}")
                            continue
                    if not matched:
                        answers_count[res] = 1
                if not answers_count:
                    majority_ans, max_count = '', 0
                else:
                    majority_ans = max(answers_count, key=answers_count.get)
                    max_count = answers_count[majority_ans]
                uid2ground_truths[uids[i]]=majority_ans
                ground_truths.append(majority_ans)
        reward_infos = []
        uid2group_acc = defaultdict(list)
        for i in range(len(data)):
            if self.num_examine > 0: # val data
                expected_gt = data[i].non_tensor_batch['reward_model'].get('ground_truth', [])
                assert expected_gt == ground_truths[i], f"val data ground_truth is not correct: expected {expected_gt}, got {ground_truths[i]}" 
            
            result = self.compute_score(responses[i], ground_truths[i])
            score: float
            valid_response_length = responses_length[i]
            if isinstance(result, dict):
                score = result["score"]
                uid2group_acc[uids[i]].append(result["acc"])
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result
                uid2group_acc[uids[i]].append(score)
                reward_extra_info["acc"].append(score)
            reward = score
            reward_tensor[i, valid_response_length - 1] = reward
            
            reward_infos.append(
                {
                    'idx':i,
                    "step": step,
                    "style": "val" if self.num_examine > 0 else "exp",
                    "question": prompts[i],
                    'raw_question': data[i].non_tensor_batch['raw_prompt'],
                    "response": responses[i],
                    "pred": result.get("pred", ""),
                    "all_labels": uid2all_labels[uids[i]],
                    'filtered_labels': uid2labels[uids[i]],
                    "ground_truth(majority)": ground_truths[i],
                    "reward": reward,
                }
            )
            
        for i, example in enumerate(reward_infos):
            acc_mean=float(np.mean(uid2group_acc[uids[i]]))
            example.update({
                'uid2group_acc':uid2group_acc[uids[i]],
                'uid2acc_mean': acc_mean,
                'is_kept': bool(acc_mean >= self.low and acc_mean <= self.high)
            })
            #print(f'{i=},{example=}')
        

        reward_extra_info['reward_infos'] = reward_infos
        #reward_infos =[example.update({'uid2group_acc':uid2group_acc[uids[i]]}) for i, example in enumerate(reward_infos)]
        #/root/users/ycy/Self-evolving-Agent/saved_results/Solver/Qwen3-4B-Base-V1
        reward_info_path_dir = f"{self.storage_path}/reward_info/"
        step_str = str(step).zfill(3)
        if self.num_examine > 0:
            os.makedirs(f"{reward_info_path_dir}/valdata", exist_ok=True)
            with open(f"{reward_info_path_dir}/valdata/step_{step_str}.jsonl", 'w', encoding='utf-8') as f:
                for reward_info in reward_infos:
                    f.write(json.dumps(reward_info, ensure_ascii=False) + '\n')
        else:
            os.makedirs(f"{reward_info_path_dir}/expdata", exist_ok=True)
            with open(f"{reward_info_path_dir}/expdata/step_{step_str}.jsonl", 'w', encoding='utf-8') as f:
                for reward_info in reward_infos:
                    f.write(json.dumps(reward_info, ensure_ascii=False) + '\n')
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

