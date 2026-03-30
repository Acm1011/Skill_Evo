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

prompt_skill_induction_v2 = """
You are a skill distiller for math reasoning. Given one question and trajectories from the same rollout group, summarize ONE reusable skill.

Question: {question}
Group trajectories:

{trajectories}

Output exactly one rule and nothing else:
WHEN <broad domain or phase>. IF <specific condition, scenario, inefficiency, or error pattern>. THEN <action to take or avoid>.

Rules:
- Group trajectories may include both [SUCCESS] and [FAIL]; use them together to infer ONE reusable skill.
- Use [SUCCESS] to identify effective strategy; use [FAIL] to identify the key mistake or missing check.
- Be generic and transferable; do not copy specific numbers.
- Keep the whole rule within 220 characters.
- Focus on improving correctness, not style.
"""

prompt_use_skill = """
SKILL: {skill}
Question: {question}
If the skills are applicable, use them to solve the problem. Please reason step by step, and put your final answer within \\boxed{{}}.
"""

prompt_use_skill_v2 = """
Please reason step by step, and put your final answer within \boxed{{}}. You will be given some reusable problem-solving skills, and you can refer to them during the reasoning.
SKILL: {skill}
Question: {question}
"""