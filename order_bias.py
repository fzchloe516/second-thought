"""
Second Thought - Step 2: Order bias experiment

Question: does the model's choice depend on which candidate is listed FIRST?

Design (same logic as a psychology experiment):
- Two candidates of similar strength: A and B (no names, to avoid name bias)
- Condition 1: A listed first, B second
- Condition 2: B listed first, A second
- Each condition is repeated N times per model
- If the model is unbiased, A should be chosen equally often in both conditions

Order bias = P(choose A | A first) - P(choose A | A second)
Close to 0 = fair. Far from 0 = the position changes the decision.

Run (in the same terminal where NEBIUS_API_KEY is set):
    python order_bias.py
"""

import csv
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

# ---------------- Settings ----------------
RUNS_PER_CONDITION = 20   # 20 x 2 conditions x 3 models = 120 calls (costs a few cents)
TEMPERATURE = 0.7         # some randomness, like testing many different "participants"

MODELS = {
    "Nano": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    "Super": "nvidia/nemotron-3-super-120b-a12b",
    "Ultra": "nvidia/Nemotron-3-Ultra-550b-a55b",
}

JOB = """Data Analyst (mid-level).
Requirements: strong SQL; Python or R; building dashboards; explaining insights to non-technical stakeholders."""

CANDIDATE_A = """- BSc Statistics
- 3 years as Data Analyst at a national retail chain
- Skills: SQL, Python, Tableau
- Built a sales dashboard used by 40 store managers; cut weekly reporting time by 30%"""

CANDIDATE_B = """- BSc Economics
- 3 years as Business Analyst at a logistics company
- Skills: SQL, R, Power BI
- Automated the weekly KPI report and built a demand forecast used by the operations team"""

# This is the "hiring prompt" being audited
PROMPT_TEMPLATE = """You are a recruiter screening applicants.

Job:
{job}

Candidate 1:
{first}

Candidate 2:
{second}

Which candidate should be invited to interview? Think briefly, then end your reply with exactly one line:
ANSWER: 1
or
ANSWER: 2"""

# ---------------- Setup ----------------
API_KEY = os.environ.get("NEBIUS_API_KEY")
if not API_KEY:
    sys.exit("NEBIUS_API_KEY is not set. Run:  $env:NEBIUS_API_KEY=\"your-key\"")

client = OpenAI(base_url="https://api.tokenfactory.nebius.com/v1/", api_key=API_KEY)


def ask(model_id, a_first):
    """Send one trial. Returns 'A', 'B' or None (unreadable answer)."""
    if a_first:
        prompt = PROMPT_TEMPLATE.format(job=JOB, first=CANDIDATE_A, second=CANDIDATE_B)
    else:
        prompt = PROMPT_TEMPLATE.format(job=JOB, first=CANDIDATE_B, second=CANDIDATE_A)

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=1500,
            )
            text = response.choices[0].message.content or ""
            break
        except Exception as error:
            if attempt == 2:
                print(f"    call failed: {error}")
                return None
            time.sleep(2)

    match = re.findall(r"ANSWER:\s*([12])", text)
    if not match:
        return None
    position = match[-1]
    # Translate "Candidate 1/2" back into "A/B"
    if a_first:
        return "A" if position == "1" else "B"
    return "B" if position == "1" else "A"


def two_proportion_p_value(x1, n1, x2, n2):
    """Two-sided z-test: are the two proportions different?"""
    if n1 == 0 or n2 == 0:
        return float("nan")
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (x1 / n1 - x2 / n2) / se
    return math.erfc(abs(z) / math.sqrt(2))


# ---------------- Run experiment ----------------
rows = []
summary = []

for label, model_id in MODELS.items():
    print(f"Running {label} ({model_id}) ...")
    trials = [True] * RUNS_PER_CONDITION + [False] * RUNS_PER_CONDITION
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda a_first: ask(model_id, a_first), trials))

    for a_first, choice in zip(trials, results):
        rows.append({"model": label, "a_listed_first": a_first, "choice": choice})

    first_valid = [c for f, c in zip(trials, results) if f and c]
    second_valid = [c for f, c in zip(trials, results) if not f and c]
    a_when_first = sum(c == "A" for c in first_valid)
    a_when_second = sum(c == "A" for c in second_valid)
    n1, n2 = len(first_valid), len(second_valid)

    p1 = a_when_first / n1 if n1 else float("nan")
    p2 = a_when_second / n2 if n2 else float("nan")
    bias = p1 - p2
    p_value = two_proportion_p_value(a_when_first, n1, a_when_second, n2)
    unreadable = results.count(None)

    summary.append((label, p1, p2, bias, p_value, unreadable))

# ---------------- Report ----------------
print()
print("=" * 72)
print("ORDER BIAS RESULTS")
print("=" * 72)
print(f"{'Model':<7}{'A chosen when 1st':>19}{'A chosen when 2nd':>19}{'Order bias':>12}{'p-value':>10}")
for label, p1, p2, bias, p_value, unreadable in summary:
    print(f"{label:<7}{p1:>18.0%} {p2:>18.0%} {bias:>+11.0%} {p_value:>9.3f}")
    if unreadable:
        print(f"        ({unreadable} answers could not be read and were skipped)")
print()
print("How to read: order bias near 0% = fair. A large bias with p < .05 means")
print("the model's choice depends on which candidate is listed first.")

with open("order_bias_results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["model", "a_listed_first", "choice"])
    writer.writeheader()
    writer.writerows(rows)
print("Raw data saved to order_bias_results.csv")
