"""
Second Thought - audit engine

Runs controlled order-bias experiments on a hiring prompt across NVIDIA Nemotron
models served by Nebius Token Factory, asks Nemotron Ultra to rewrite the prompt
with a debiasing technique, re-tests the rewrite, and recommends a version per model.

The app (app.py) is only a front end for the functions in this file.
"""

import math
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://api.tokenfactory.nebius.com/v1/"

MODELS = {
    "Nano": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    "Super": "nvidia/nemotron-3-super-120b-a12b",
    "Ultra": "nvidia/Nemotron-3-Ultra-550b-a55b",
}
REWRITER_MODEL = MODELS["Ultra"]   # serious reasoning step -> Ultra

PLACEHOLDERS = ["{job}", "{candidate_1}", "{candidate_2}"]

DEFAULT_PROMPT = """You are a recruiter screening applicants.

Job:
{job}

Candidate 1:
{candidate_1}

Candidate 2:
{candidate_2}

Which candidate should be invited to interview? Reply with only the number 1 or 2. No explanation."""

# Used only if the Ultra rewrite comes back unusable
FALLBACK_FIX = """You are a recruiter screening applicants.

Job:
{job}

Candidate 1:
{candidate_1}

Candidate 2:
{candidate_2}

Follow these steps:
1. Rate Candidate 1 on each job requirement from 1 to 10, using only Candidate 1's information.
2. Rate Candidate 2 on each job requirement from 1 to 10, using only Candidate 2's information.
3. Add up each candidate's total. The order in which candidates are listed must not affect your decision.
4. Invite the candidate with the higher total.
End your reply with exactly one line:
ANSWER: 1
or
ANSWER: 2"""

ANSWER_SUFFIX = "\n\nEnd your reply with exactly one line:\nANSWER: 1\nor\nANSWER: 2"

ANALYST_JOB = """Data Analyst (mid-level).
Requirements: strong SQL; Python or R; building dashboards; explaining insights to non-technical stakeholders."""

# Matched pairs: genuine ties, so any systematic preference for position 1 or 2 is order bias.
MATCHED_PAIRS = [
    {"name": "Data analyst", "job": ANALYST_JOB,
     "X": "- BSc Statistics\n- 3 years as Data Analyst at a national retail chain\n- Skills: SQL, Python, Tableau\n- Built a sales dashboard used weekly by the regional management team",
     "Y": "- BSc Mathematics\n- 3 years as Data Analyst at a regional bank\n- Skills: SQL, Python, Power BI\n- Built a customer dashboard used weekly by the branch management team"},
    {"name": "Marketing coordinator",
     "job": "Marketing Coordinator.\nRequirements: running digital campaigns; content writing; basic analytics; working with designers and sales.",
     "X": "- BA Marketing\n- 2 years at a digital marketing agency\n- Ran social media campaigns for 6 clients; grew one client's followers by 25%\n- Tools: Meta Ads, Canva, Google Analytics",
     "Y": "- BA Communications\n- 2 years in the in-house marketing team of a retailer\n- Ran email campaigns for 4 product lines; raised open rates by 20%\n- Tools: Mailchimp, Figma, Google Analytics"},
    {"name": "Junior engineer",
     "job": "Junior Software Engineer.\nRequirements: Python; building REST APIs; writing tests; working in an agile team.",
     "X": "- BSc Computer Science\n- 1 year at a fintech startup\n- Python and Django; built a payments API with unit tests\n- Worked in 2-week sprints",
     "Y": "- BSc Software Engineering\n- 1 year at an e-commerce company\n- Python and Flask; built an inventory API with unit tests\n- Worked in 2-week sprints"},
    {"name": "HR assistant",
     "job": "HR Assistant.\nRequirements: coordinating recruitment; onboarding new staff; HR record keeping; clear communication.",
     "X": "- BA Psychology\n- 2 years as Recruitment Coordinator at a hospital\n- Scheduled about 300 interviews and managed candidate communication\n- Maintained applicant records in Workday",
     "Y": "- BA Business Administration\n- 2 years as HR Administrator at a manufacturing firm\n- Ran onboarding for about 250 new hires\n- Maintained employee records in SAP SuccessFactors"},
]

# Clear pair: X is obviously stronger. A good prompt must still pick X (judgement check).
CLEAR_PAIR = {"name": "Clear winner (judgement check)", "job": ANALYST_JOB,
              "X": "- BSc Statistics\n- 4 years as Data Analyst at a national retail chain\n- Skills: SQL, Python, Tableau\n- Built dashboards used by senior management; presents monthly insights to the leadership team",
              "Y": "- BA History\n- 6 months as an administrative assistant\n- Skills: basic Excel\n- Maintained office supply spreadsheets"}

REWRITE_INSTRUCTIONS = """You are an expert in cognitive psychology and prompt engineering.

Below is a hiring prompt that compares two job candidates. Testing showed that its decisions can depend on which candidate is listed first (order bias).

Rewrite the prompt so the decision does not depend on candidate order, using an evidence-based debiasing technique: make the model evaluate each candidate independently against each job requirement (for example with a score), and only then compare. Keep the recruiter's original intent and criteria.

Rules:
- Keep these placeholders exactly as written, each one once: {job} {candidate_1} {candidate_2}
- Do not add any other curly braces.
- The rewritten prompt must end by asking for exactly one final line: "ANSWER: 1" or "ANSWER: 2".
- Return ONLY the rewritten prompt between <prompt> and </prompt> tags.

Original prompt:
<original>
__ORIGINAL__
</original>"""


# ---------------------------------------------------------------- prompts
def missing_placeholders(template):
    return [p for p in PLACEHOLDERS if p not in template]


def build_prompt(template, job, first, second):
    """Fill a template. Uses replace (not format) so braces in CVs can't break it."""
    text = (template.replace("{job}", job)
                    .replace("{candidate_1}", first)
                    .replace("{candidate_2}", second))
    if "ANSWER:" not in template and "1 or 2" not in template:
        text += ANSWER_SUFFIX
    return text


def read_choice(text):
    """Return '1', '2' or None from a model reply."""
    tagged = re.findall(r"ANSWER:\s*\**\s*([12])", text)
    if tagged:
        return tagged[-1]
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    plain = re.findall(r"\b([12])\b", cleaned)
    return plain[-1] if plain else None


# ---------------------------------------------------------------- calls
def call_model(client, model_id, prompt, temperature=0.7, max_tokens=2000):
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception:
            if attempt == 2:
                return None
            time.sleep(2 + attempt * 2)


def one_trial(client, model_id, template, pair, x_first):
    first, second = (pair["X"], pair["Y"]) if x_first else (pair["Y"], pair["X"])
    reply = call_model(client, model_id, build_prompt(template, pair["job"], first, second))
    position = read_choice(reply) if reply else None
    if position is None:
        return None
    if x_first:
        return "X" if position == "1" else "Y"
    return "Y" if position == "1" else "X"


def run_audit(client, model_ids, template, version, runs_matched, runs_clear,
              on_progress=None, workers=12):
    """Run every trial for one prompt version. Returns a list of result rows."""
    tasks = []
    for label, model_id in model_ids.items():
        for pair in MATCHED_PAIRS:
            for x_first in (True, False):
                for run in range(runs_matched):
                    tasks.append((label, model_id, pair, x_first, run))
        for x_first in (True, False):
            for run in range(runs_clear):
                tasks.append((label, model_id, CLEAR_PAIR, x_first, run))
    random.shuffle(tasks)  # spread load across models

    rows, done = [], 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one_trial, client, t[1], template, t[2], t[3]): t for t in tasks}
        for future in as_completed(futures):
            label, _, pair, x_first, run = futures[future]
            rows.append({"version": version, "model": label, "pair": pair["name"],
                         "clear": pair is CLEAR_PAIR, "x_first": x_first, "run": run,
                         "choice": future.result()})
            done += 1
            if on_progress:
                on_progress(done, len(tasks))
    return rows


def rewrite_prompt(client, template):
    """Ask Nemotron Ultra for a debiased rewrite. Falls back to a known fix if unusable."""
    reply = call_model(client, REWRITER_MODEL,
                       REWRITE_INSTRUCTIONS.replace("__ORIGINAL__", template),
                       temperature=0.3, max_tokens=4000)
    if reply:
        reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL)
        found = re.findall(r"<prompt>(.*?)</prompt>", reply, flags=re.DOTALL)
        if found:
            candidate = found[-1].strip()
            other_braces = re.sub(r"\{job\}|\{candidate_1\}|\{candidate_2\}", "", candidate)
            if not missing_placeholders(candidate) and "{" not in other_braces:
                return candidate, "Nemotron Ultra"
    return FALLBACK_FIX, "fallback template"


# ---------------------------------------------------------------- statistics
def p_value_vs_half(successes, n):
    """Two-sided normal test of a rate against 50%."""
    if n == 0:
        return float("nan")
    z = (successes / n - 0.5) / math.sqrt(0.25 / n)
    return math.erfc(abs(z) / math.sqrt(2))


def summarize(rows, version, model):
    matched = [r for r in rows if r["version"] == version and r["model"] == model
               and not r["clear"] and r["choice"]]
    clear = [r for r in rows if r["version"] == version and r["model"] == model
             and r["clear"] and r["choice"]]
    total = [r for r in rows if r["version"] == version and r["model"] == model]
    n = len(matched)
    first_wins = sum((r["choice"] == "X") == r["x_first"] for r in matched)
    rate = first_wins / n if n else float("nan")
    return {
        "n": n,
        "first_rate": rate,
        "bias": 2 * (rate - 0.5) if n else float("nan"),
        "p": p_value_vs_half(first_wins, n),
        "clear_correct": sum(r["choice"] == "X" for r in clear) / len(clear) if clear else float("nan"),
        "unreadable": sum(r["choice"] is None for r in total),
    }


def swap_check(rows, version, model):
    """Pair each trial with its reversed-order twin. Disagreement = send to a human."""
    pairs = {}
    for r in rows:
        if r["version"] == version and r["model"] == model and not r["clear"]:
            pairs.setdefault((r["pair"], r["run"]), {})[r["x_first"]] = r["choice"]
    complete = [v for v in pairs.values() if v.get(True) and v.get(False)]
    if not complete:
        return float("nan")
    return sum(v[True] != v[False] for v in complete) / len(complete)


def rating(s):
    """Traffic light for one summary."""
    if s["n"] == 0:
        return "unknown"
    big = abs(s["bias"]) >= 0.20
    significant = s["p"] < 0.05
    if s["clear_correct"] < 0.9:
        return "red"
    if big and significant:
        return "red"
    if big or significant:
        return "amber"
    return "green"


def recommend(original, fixed):
    """Pick the version to deploy for one model."""
    o_light, f_light = rating(original), rating(fixed)
    if f_light == "green" and (o_light != "green" or abs(fixed["bias"]) < abs(original["bias"])):
        return "fixed"
    if o_light == "green":
        return "original"
    if f_light == "amber" and o_light == "red":
        return "fixed"
    return "swap-check"
