prompt_skill_induction = """
You are a skill distiller for math reasoning. Given one question and trajectories from the same rollout group, summarize ONE reusable skill.
Question: {question}
Group trajectories:

{trajectories}

Output MUST be a valid JSON object and nothing else (no markdown/code fences).
Schema: {{"skill name", "problem type", "key insight", "method"}}

Rules:
- Group trajectories may include both [SUCCESS] and [FAIL]; infer ONE reusable skill from them, where [SUCCESS] is correct and [FAIL] is incorrect.
- Be generic and transferable; do not copy specific numbers. 
- Keep the whole skill within 220 characters. 
- key insight is the most important field. 
- method must contain 2--3 concise steps. 
- Focus on improving correctness, not style.
"""

prompt_use_skill = """
SKILL: {skill}
Question: {question}
If the skills are applicable, use them to solve the problem. Please reason step by step, and put your final answer within \\boxed{{}}.
"""