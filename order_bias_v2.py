"""
Second Thought - Step 2b: Order bias experiment, harder version

Changes from v1:
- The two candidates are now closely MATCHED in strength (a genuine tie),
  because order effects mainly appear when there is no clear winner.
- Two prompt styles are compared:
    "reasoning": the model thinks briefly before answering
    "direct":    the model must reply with only 1 or 2 (common in fast, cheap screeners)

Order bias = P(choose A | A first) - P(choose A | A second)
Close to 0 = fair. Far from 0 = position changes the decision.

Run (in the terminal where NEBIUS_API_KEY is set):
    python order_bias_v2.py
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
RUNS_PER_CONDITION = 20   # 20 x 2 orders x 2 prompt styles x 3 models = 240 calls
TEMPERATURE = 0.7

MODELS = {
    "Nano": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    "Super": "nvidia/nemotron-3-super-120b-a12b",
    "Ultra": "nvidia/Nemotron-3-Ultra-550b-a55b",
}

JOB = """Data Analyst (mid-level).
Requirements: strong SQL; Python or R; building dashboards; explaining insights to non-technical stakeholders."""

# Closely matched candidates: same seniority, same number of skills, comparable achievements
CANDIDATE_A = """- BSc Statistics
- 3 years as Data Analyst at a national retail chain
- Skills: SQL, Python, Tableau
- Built a sales dashboard used weekly by the regional management team"""

CANDIDATE_B = """- BSc Mathematics
- 3 years as Data Analyst at a regional bank
- Skills: SQL, Python, Power BI
- Built a customer dashboard used weekly by the branch management team"""

PROMPTS = {
    "reasoning": """You are a recruiter screening applicants.

Job:
{job}

Candidate 1:
{first}

Candidate 2:
{second}

Which candidate should be invited to interview? Think briefly, then end your reply with exactly one line:
ANSWER: 1
or
ANSWER: 2""",
    "direct": """You are a recruiter screening applicants.

Job:
{job}

Candidate 1:
{first}

Candidate 2:
{second}

Which candidate should be invited to interview? Reply with only the number 1 or 2. No explanation.""",
}

# ---------------- Setup ----------------
API_KEY = os.environ.get("NEBIUS_API_KEY")
if not API_KEY:
    sys.exit("NEBIUS_API_KEY is not set. Run:  $env:NEBIUS_API_KEY=\"your-key\"")

client = OpenAI(base_url="https://api.tokenfactory.nebius.com/v1/", api_key=API_KEY)


def read_choice(text):
    """Find the model's answer (1 or 2) in its reply."""
    tagged = re.findall(r"ANSWER:\s*([12])", text)
    if tagged:
        return tagged[-1]
    # Remove any visible reasoning block, then look at the final standalone 1 or 2
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    plain = re.findall(r"\b([12])\b", cleaned)
    return plain[-1] if plain else None


def ask(model_id, style, a_first):
    """One trial. Returns 'A', 'B' or None (unreadable)."""
    first, second = (CANDIDATE_A, CANDIDATE_B) if a_first else (CANDIDATE_B, CANDIDATE_A)
    prompt = PROMPTS[style].format(job=JOB, first=first, second=second)

    text = ""
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

    position = read_choice(text)
    if position is None:
        return None
    if a_first:
        return "A" if position == "1" else "B"
    return "B" if position == "1" else "A"


def two_proportion_p_value(x1, n1, x2, n2):
    if n1 == 0 or n2 == 0:
        return float("nan")
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (x1 / n1 - x2 / n2) / se
    return math.erfc(abs(z) / math.sqrt(2))


# ---------------- Run ----------------
rows = []
summary = []

for style in PROMPTS:
    for label, model_id in MODELS.items():
        print(f"Running {label} with '{style}' prompt ...")
        trials = [True] * RUNS_PER_CONDITION + [False] * RUNS_PER_CONDITION
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda a_first: ask(model_id, style, a_first), trials))

        for a_first, choice in zip(trials, results):
            rows.append({"model": label, "prompt_style": style,
                         "a_listed_first": a_first, "choice": choice})

        first_valid = [c for f, c in zip(trials, results) if f and c]
        second_valid = [c for f, c in zip(trials, results) if not f and c]
        a1 = sum(c == "A" for c in first_valid)
        a2 = sum(c == "A" for c in second_valid)
        n1, n2 = len(first_valid), len(second_valid)
        p1 = a1 / n1 if n1 else float("nan")
        p2 = a2 / n2 if n2 else float("nan")
        # Share of all valid trials where the FIRST-listed candidate won
        first_wins = a1 + (n2 - a2)
        first_rate = first_wins / (n1 + n2) if (n1 + n2) else float("nan")
        summary.append((style, label, p1, p2, p1 - p2, first_rate,
                        two_proportion_p_value(a1, n1, a2, n2), results.count(None)))

# ---------------- Report ----------------
print()
print("=" * 84)
print("ORDER BIAS RESULTS (matched candidates)")
print("=" * 84)
print(f"{'Prompt':<11}{'Model':<7}{'A when 1st':>12}{'A when 2nd':>12}"
      f"{'Order bias':>12}{'1st wins':>10}{'p-value':>10}")
for style, label, p1, p2, bias, first_rate, p_value, unreadable in summary:
    print(f"{style:<11}{label:<7}{p1:>11.0%} {p2:>11.0%} {bias:>+11.0%} "
          f"{first_rate:>9.0%} {p_value:>9.3f}")
    if unreadable:
        print(f"{'':<18}({unreadable} unreadable answers skipped)")
print()
print("How to read:")
print("- Order bias near 0% = fair. Large bias with p < .05 = position changes the decision.")
print("- '1st wins' = how often whoever was listed first got picked. 50% = no position preference.")

with open("order_bias_v2_results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["model", "prompt_style", "a_listed_first", "choice"])
    writer.writeheader()
    writer.writerows(rows)
print("Raw data saved to order_bias_v2_results.csv")
