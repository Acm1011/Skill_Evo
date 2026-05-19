MATH_TEMPLATE_NO_HIS = """
You are solving a mathematics problem.
{reflections}
Problem:
{question}

Respond for this attempt as follows:
1. Put your reasoning inside <think> and </think>.
2. Put the final answer only inside <answer> and </answer>.
3. Do not omit the <answer> tag.
"""


MATH_TEMPLATE = """
You are solving a mathematics problem.
{reflections}
Problem:
{question}

You have already made {step_count} prior attempt record(s). Recent history:
{memory_context}

Respond for this attempt as follows:
1. Put your reasoning inside <think> and </think>.
2. Put the final answer only inside <answer> and </answer>.
3. Do not omit the <answer> tag.
"""


MATH_REFLECT_TEMPLATE = """
You are reflecting on a mathematics problem attempt.
Problem:
{question}

{reference_trajectory}

Current trajectory:
{current_trajectory}

Analyze the attempt and then output JSON only.

Required JSON schema:
{{
  "subtasks": [
    {{"name": "understand_problem", "description": "...", "status": "completed or incomplete"}},
    {{"name": "choose_strategy", "description": "...", "status": "completed or incomplete"}},
    {{"name": "derive_solution", "description": "...", "status": "completed or incomplete"}},
    {{"name": "check_constraints", "description": "...", "status": "completed or incomplete"}},
    {{"name": "finalize_answer", "description": "...", "status": "completed or incomplete"}}
  ],
  "task_success": true,
  "action_lesson": "...",
  "reasoning_lesson": "..."
}}

Requirements:
- Use exactly the five subtask names above.
- Set "task_success" to whether the attempt really solved the problem.
- "action_lesson" should capture the most important strategic takeaway.
- "reasoning_lesson" should capture the most important derivation or verification takeaway.
- If a lesson is unavailable, use an empty string.
- Output JSON only.
"""

