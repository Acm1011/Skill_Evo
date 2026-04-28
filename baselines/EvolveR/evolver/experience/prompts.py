"""
LLM Prompts for the Experience Manager (math tasks; baselines/EvolveR).
"""

DESCRIPTION_PART_SEPARATOR = "[DESCRIPTION]:"

STRUCTURED_PART_SEPARATOR = "[STRUCTURE]:"

SUMMARIZE_SUCCESSFUL_TRAJECTORY_PROMPT = f"""
You analyze math problem-solving interaction logs and distill a short "Guiding Principle"
that could help on similar problems later.

A Guiding Principle has two parts:
1. One concise sentence: the core strategy or insight.
2. A list of simple (subject, predicate, object) triplets capturing the reasoning pattern.

[Trajectory Log]:
{{trajectory_log}}

Final Outcome: SUCCESS

**Your Task:**
Generate the Guiding Principle.
First, on a new line, write `{DESCRIPTION_PART_SEPARATOR}`.
Then, the one-sentence description of the principle.
Then, on a new line, write `{STRUCTURED_PART_SEPARATOR}`.
Finally, structured triplets as a valid JSON list (use strings for each triplet if needed).

[Example]:
{DESCRIPTION_PART_SEPARATOR}
When solving a quadratic, complete the square before taking roots if the linear term dominates.
{STRUCTURED_PART_SEPARATOR}
[
  ("quadratic_equation", "method", "complete_the_square"),
  ("taking_roots", "after", "canonical_form")
]

[Output]:
"""

SUMMARIZE_FAILED_TRAJECTORY_PROMPT = f"""
You analyze math problem-solving logs to extract a "Cautionary Principle" — a mistake to avoid.

A Cautionary Principle has two parts:
1. One concise sentence: what went wrong and when to watch for it.
2. Triplets describing the failure pattern.

[Trajectory Log]:
{{trajectory_log}}

Final Outcome: FAILURE

**Your Task:**
Generate the Cautionary Principle.
First, on a new line, write `{DESCRIPTION_PART_SEPARATOR}`.
Then, the one-sentence description of the pitfall.
Then, on a new line, write `{STRUCTURED_PART_SEPARATOR}`.
Finally, structured triplets in a valid JSON list format.

[Example]:
{DESCRIPTION_PART_SEPARATOR}
Rushing to differentiate without checking domain constraints led to an extraneous critical point.
{STRUCTURED_PART_SEPARATOR}
[
  ("differentiation", "skipped", "domain_check"),
  ("critical_point", "invalid", "boundary_violation")
]

[Output]:
"""


MATCH_PRINCIPLE_PROMPT = """
You are a semantic analysis expert. Determine if two principles describe the same core idea, even if they use different words.

Principle A: "{summary}"
Principle B: "{existing_principle_description}"

Do Principle A and Principle B describe the same essential advice or warning?
Please answer with only "Yes" or "No".
"""
