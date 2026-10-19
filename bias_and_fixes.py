"""
Second Thought - Step 3: Confirm order bias across several candidate pairs, and test two fixes

Candidate pairs
- 4 MATCHED pairs (genuine ties, different jobs) -> measure order bias
- 1 CLEAR pair (one candidate is obviously stronger) -> judgement check

Conditions
- "direct": reply with only 1 or 2 (baseline, the risky style)
- "scoring": FIX A, score each candidate independently on each requirement, then compare
- "swap-check": FIX B, a procedure, no extra calls. Pair each "direct" trial with its
  reversed-order twin. If both picks agree -> accept. If they disagree -> "tie, send to a human".

Key measure: "1st wins" = how often whoever is listed first gets picked.
50% = no position effect. Order bias = 2 x (1st wins - 50%).

Run (in the terminal where NEBIUS_API_KEY is set):
    python bias_and_fixes.py
Takes roughly 5-15 minutes. Costs well under $2.
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
RUNS_MATCHED = 15   # per pair, per order, per condition, per model
RUNS_CLEAR = 10
TEMPERATURE = 0.7
WORKERS = 12

MODELS = {
    "Nano": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    "Super": "nvidia/nemotron-3-super-120b-a12b",
    "Ultra": "nvidia/Nemotron-3-Ultra-550b-a55b",
}

ANALYST_JOB = """Data Analyst (mid-level).
Requirements: strong SQL; Python or R; building dashboards; explaining insights to non-technical stakeholders."""

MATCHED_PAIRS = [
    {
        "name": "analyst",
        "job": ANALYST_JOB,
        "X": """- BSc Statistics
- 3 years as Data Analyst at a national retail chain
- Skills: SQL, Python, Tableau
- Built a sales dashboard used weekly by the regional management team""",
        "Y": """- BSc Mathematics
- 3 years as Data Analyst at a regional bank
- Skills: SQL, Python, Power BI
- Built a customer dashboard used weekly by the branch management team""",
    },
    {
        "name": "marketing",
        "job": """Marketing Coordinator.
Requirements: running digital campaigns; content writing; basic analytics; working with designers and sales.""",
        "X": """- BA Marketing
- 2 years at a digital marketing agency
- Ran social media campaigns for 6 clients; grew one client's followers by 25%
- Tools: Meta Ads, Canva, Google Analytics""",
        "Y": """- BA Communications
- 2 years in the in-house marketing team of a retailer
- Ran email campaigns for 4 product lines; raised open rates by 20%
- Tools: Mailchimp, Figma, Google Analytics""",
    },
    {
        "name": "engineer",
        "job": """Junior Software Engineer.
Requirements: Python; building REST APIs; writing tests; working in an agile team.""",
        "X": """- BSc Computer Science
- 1 year at a fintech startup
- Python and Django; built a payments API with unit tests
- Worked in 2-week sprints""",
        "Y": """- BSc Software Engineering
- 1 year at an e-commerce company
- Python and Flask; built an inventory API with unit tests
- Worked in 2-week sprints""",
    },
    {
        "name": "hr",
        "job": """HR Assistant.
Requirements: coordinating recruitment; onboarding new staff; HR record keeping; clear communication.""",
        "X": """- BA Psychology
- 2 years as Recruitment Coordinator at a hospital
- Scheduled about 300 interviews and managed candidate communication
- Maintained applicant records in Workday""",
        "Y": """- BA Business Administration
- 2 years as HR Administrator at a manufacturing firm
- Ran onboarding for about 250 new hires
- Maintained employee records in SAP SuccessFactors""",
    },
]

CLEAR_PAIR = {
    "name": "clear",
    "job": ANALYST_JOB,
    "X": """- BSc Statistics
- 4 years as Data Analyst at a national retail chain
- Skills: SQL, Python, Tableau
- Built dashboards used by senior management; presents monthly insights to the leadership team""",
    "Y": """- BA History
- 6 months as an administrative assistant
- Skills: basic Excel
- Maintained office supply spreadsheets""",
}  # X is clearly stronger

PROMPTS = {
    "direct": """You are a recruiter screening applicants.

Job:
{job}

Candidate 1:
{first}

Candidate 2:
{second}

Which candidate should be invited to interview? Reply with only the number 1 or 2. No explanation.""",
    "scoring": """You are a recruiter screening applicants.

Job:
{job}

Candidate 1:
{first}

Candidate 2:
{second}

Follow these steps:
1. Rate Candidate 1 on each job requirement from 1 to 10, using only Candidate 1's information.
2. Rate Candidate 2 on each job requirement from 1 to 10, using only Candidate 2's information.
3. Add up each candidate's total. The order in which candidates are listed must not affect your decision.
4. Invite the candidate with the higher total.
End your reply with exactly one line:
ANSWER: 1
or
ANSWER: 2""",
}

# ---------------- Setup ----------------
API_KEY = os.environ.get("NEBIUS_API_KEY")
if not API_KEY:
    sys.exit("NEBIUS_API_KEY is not set. Run:  $env:NEBIUS_API_KEY=\"your-key\"")

client = OpenAI(base_url="https://api.tokenfactory.nebius.com/v1/", api_key=API_KEY)


def read_choice(text):
    tagged = re.findall(r"ANSWER:\s*([12])", text)
    if tagged:
        return tagged[-1]
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    plain = re.findall(r"\b([12])\b", cleaned)
    return plain[-1] if plain else None


def ask(task):
    """task = (model_id, style, pair, x_first). Returns 'X', 'Y' or None."""
    model_id, style, pair, x_first = task
    first, second = (pair["X"], pair["Y"]) if x_first else (pair["Y"], pair["X"])
    prompt = PROMPTS[style].format(job=pair["job"], first=first, second=second)
    text = ""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=2000,
            )
            text = response.choices[0].message.content or ""
            break
        except Exception as error:
            if attempt == 2:
                print(f"    call failed: {error}")
                return None
            time.sleep(3)
    position = read_choice(text)
    if position is None:
        return None
    if x_first:
        return "X" if position == "1" else "Y"
    return "Y" if position == "1" else "X"


def p_value_vs_half(successes, n):
    """Two-sided test: is this rate different from 50%?"""
    if n == 0:
        return float("nan")
    z = (successes / n - 0.5) / math.sqrt(0.25 / n)
    return math.erfc(abs(z) / math.sqrt(2))


# ---------------- Build and run all trials ----------------
rows = []
for label, model_id in MODELS.items():
    tasks = []
    for style in PROMPTS:
        for pair in MATCHED_PAIRS:
            for x_first in (True, False):
                for run in range(RUNS_MATCHED):
                    tasks.append((model_id, style, pair, x_first, run))
        for x_first in (True, False):
            for run in range(RUNS_CLEAR):
                tasks.append((model_id, style, CLEAR_PAIR, x_first, run))

    print(f"Running {label}: {len(tasks)} trials ...")
    started = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        choices = list(pool.map(lambda t: ask(t[:4]), tasks))
    print(f"  done in {time.time() - started:.0f}s")

    for (model_id_, style, pair, x_first, run), choice in zip(tasks, choices):
        rows.append({"model": label, "style": style, "pair": pair["name"],
                     "x_first": x_first, "run": run, "choice": choice})


# ---------------- Analyse ----------------
def select(model, style, pair_names):
    return [r for r in rows if r["model"] == model and r["style"] == style
            and r["pair"] in pair_names and r["choice"] is not None]


matched_names = [p["name"] for p in MATCHED_PAIRS]
print()
print("=" * 86)
print("RESULTS: order bias on 4 matched pairs + judgement check on 1 clear pair")
print("=" * 86)
print(f"{'Model':<7}{'Method':<12}{'1st wins':>10}{'Order bias':>12}{'p-value':>10}"
      f"{'Clear pair correct':>20}{'Sent to human':>15}")

for label in MODELS:
    for style in PROMPTS:
        data = select(label, style, matched_names)
        first_wins = sum((r["choice"] == "X") == r["x_first"] for r in data)
        n = len(data)
        rate = first_wins / n if n else float("nan")
        clear = select(label, style, ["clear"])
        correct = sum(r["choice"] == "X" for r in clear) / len(clear) if clear else float("nan")
        print(f"{label:<7}{style:<12}{rate:>9.0%} {2 * (rate - 0.5):>+11.0%} "
              f"{p_value_vs_half(first_wins, n):>9.3f} {correct:>19.0%} {'-':>14}")

    # FIX B: swap-check, built from the "direct" trials
    def swap_check(pair_names):
        decided, ties, picks = 0, 0, []
        for pair_name in pair_names:
            for run in range(RUNS_MATCHED if pair_name != "clear" else RUNS_CLEAR):
                a = [r for r in rows if r["model"] == label and r["style"] == "direct"
                     and r["pair"] == pair_name and r["run"] == run and r["x_first"]]
                b = [r for r in rows if r["model"] == label and r["style"] == "direct"
                     and r["pair"] == pair_name and r["run"] == run and not r["x_first"]]
                if not a or not b or a[0]["choice"] is None or b[0]["choice"] is None:
                    continue
                if a[0]["choice"] == b[0]["choice"]:
                    decided += 1
                    picks.append(a[0]["choice"])
                else:
                    ties += 1
        return decided, ties, picks

    _, ties_m, _ = swap_check(matched_names)
    decided_c, ties_c, picks_c = swap_check(["clear"])
    total_m = RUNS_MATCHED * len(matched_names)
    clear_correct = picks_c.count("X") / (decided_c + ties_c) if (decided_c + ties_c) else float("nan")
    print(f"{label:<7}{'swap-check':<12}{'n/a':>9} {'0 (design)':>11} {'-':>9} "
          f"{clear_correct:>19.0%} {ties_m / total_m:>14.0%}")
    print()

print("How to read:")
print("- 1st wins: how often the first-listed candidate was picked. 50% = no order effect.")
print("- Order bias near 0% with p > .05 = fair. Large bias with p < .05 = position matters.")
print("- Clear pair correct: did it still pick the obviously stronger candidate? (should be ~100%)")
print("- Sent to human (swap-check only): share of tie cases where the two orders disagreed.")

with open("bias_and_fixes_results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["model", "style", "pair", "x_first", "run", "choice"])
    writer.writeheader()
    writer.writerows(rows)
print("Raw data saved to bias_and_fixes_results.csv")
