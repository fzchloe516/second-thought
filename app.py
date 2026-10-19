"""
Second Thought - Streamlit front end.

Run locally:
    pip install -r requirements.txt
    $env:NEBIUS_API_KEY="your-key"        (Windows PowerShell)
    streamlit run app.py
"""

import os

import altair as alt
import pandas as pd
import streamlit as st
from openai import OpenAI

import audit_engine as eng

st.set_page_config(page_title="Second Thought", page_icon="🧠", layout="wide")

MAX_AUDITS_PER_SESSION = 3
COLOR_ORIGINAL = "#eb6834"   # validated categorical pair (CVD-safe)
COLOR_FIXED = "#2a78d6"
LIGHTS = {
    "green": ("🟢", "Fair"),
    "amber": ("🟡", "Watch"),
    "red": ("🔴", "Biased"),
    "unknown": ("⚪", "No data"),
}
ADVICE = {
    "fixed": "Use the **rewritten prompt** on this model.",
    "original": "Keep your **original prompt** on this model. The rewrite did not help here.",
    "swap-check": "Neither version is reliable here. Ask twice with the candidates swapped "
                  "and send disagreements to a human reviewer.",
}


# ---------------------------------------------------------------- API key
def get_api_key():
    try:
        if "NEBIUS_API_KEY" in st.secrets:
            return st.secrets["NEBIUS_API_KEY"]
    except Exception:
        pass
    return os.environ.get("NEBIUS_API_KEY") or st.session_state.get("user_key")


def has_server_key():
    """True when the key comes from deployment secrets or the environment."""
    try:
        if "NEBIUS_API_KEY" in st.secrets:
            return True
    except Exception:
        pass
    return bool(os.environ.get("NEBIUS_API_KEY"))


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    if not has_server_key():
        st.session_state["user_key"] = st.text_input(
            "Nebius Token Factory API key", type="password",
            help="Only kept in this browser session. Never stored.")
    chosen = st.multiselect("Nemotron models to audit", list(eng.MODELS),
                            default=list(eng.MODELS))
    runs = st.slider("Trials per candidate order", 3, 15, 5,
                     help="More trials = more reliable statistics, but slower.")
    runs_clear = 3
    st.caption(f"Each audit sends about "
               f"{2 * len(chosen) * (len(eng.MATCHED_PAIRS) * 2 * runs + 2 * runs_clear) + 1} "
               "calls to Nebius Token Factory.")
    st.divider()
    st.markdown(
        "**How it works**\n\n"
        "1. Runs your prompt on matched candidate pairs, in both orders\n"
        "2. Measures how often whoever is listed **first** wins (fair = 50%)\n"
        "3. Nemotron Ultra rewrites your prompt to evaluate candidates independently\n"
        "4. Re-tests the rewrite and checks it still picks the clearly stronger candidate\n"
        "5. Recommends a version for each model")

# ---------------------------------------------------------------- header
st.title("🧠 Second Thought")
st.markdown("#### Catch cognitive bias in your AI hiring prompt, then fix it with proof.")
st.markdown(
    "When two applicants are equally strong, an AI screener should pick each about half the time, "
    "no matter who is listed first. Second Thought runs controlled experiments from psychology on "
    "your prompt across **NVIDIA Nemotron** models served by **Nebius Token Factory**.")

# ---------------------------------------------------------------- input
st.subheader("1. Paste your hiring prompt")
template = st.text_area(
    "Use the placeholders {job}, {candidate_1} and {candidate_2} where your system inserts them.",
    value=eng.DEFAULT_PROMPT, height=260)
missing = eng.missing_placeholders(template)
if missing:
    st.error("Your prompt is missing: " + ", ".join(missing))

with st.expander("See the test candidates (4 matched pairs + 1 clear-winner check)"):
    for pair in eng.MATCHED_PAIRS + [eng.CLEAR_PAIR]:
        st.markdown(f"**{pair['name']}**  \n*Job:* {pair['job'].splitlines()[0]}")
        left, right = st.columns(2)
        left.text(pair["X"])
        right.text(pair["Y"])

st.subheader("2. Run the audit")
audits_used = st.session_state.get("audits_used", 0)
api_key = get_api_key()
ready = bool(api_key) and not missing and bool(chosen) and audits_used < MAX_AUDITS_PER_SESSION
if not api_key:
    st.info("Add a Nebius Token Factory API key in the sidebar to run an audit.")
if audits_used >= MAX_AUDITS_PER_SESSION:
    st.warning("Audit limit for this session reached. Refresh the page to start again.")

if st.button("Run bias audit", type="primary", disabled=not ready):
    st.session_state["audits_used"] = audits_used + 1
    client = OpenAI(base_url=eng.BASE_URL, api_key=api_key)
    model_ids = {k: eng.MODELS[k] for k in chosen}

    with st.status("Running audit ...", expanded=True) as status:
        st.write("**Step 1/3:** testing your original prompt")
        bar1 = st.progress(0.0)
        rows = eng.run_audit(client, model_ids, template, "original", runs, runs_clear,
                             on_progress=lambda d, t: bar1.progress(d / t, f"{d}/{t} decisions"))

        st.write("**Step 2/3:** Nemotron Ultra is rewriting your prompt")
        fixed_template, author = eng.rewrite_prompt(client, template)

        st.write("**Step 3/3:** re-testing the rewritten prompt")
        bar2 = st.progress(0.0)
        rows += eng.run_audit(client, model_ids, fixed_template, "fixed", runs, runs_clear,
                              on_progress=lambda d, t: bar2.progress(d / t, f"{d}/{t} decisions"))
        status.update(label="Audit complete", state="complete", expanded=False)

    st.session_state["result"] = {"rows": rows, "fixed": fixed_template, "author": author,
                                  "models": chosen, "original": template}

# ---------------------------------------------------------------- report
result = st.session_state.get("result")
if result:
    rows = result["rows"]
    st.subheader("3. Bias check-up report")

    table = []
    for model in result["models"]:
        for version in ("original", "fixed"):
            s = eng.summarize(rows, version, model)
            table.append({"Model": model,
                          "Prompt": "Original" if version == "original" else "Rewritten",
                          "version": version, **s,
                          "light": eng.rating(s),
                          "human": eng.swap_check(rows, version, model)})
    df = pd.DataFrame(table)

    # --- one card per model
    cols = st.columns(len(result["models"]))
    for col, model in zip(cols, result["models"]):
        o = df[(df.Model == model) & (df.version == "original")].iloc[0]
        f = df[(df.Model == model) & (df.version == "fixed")].iloc[0]
        choice = eng.recommend(o.to_dict(), f.to_dict())
        with col.container(border=True):
            st.markdown(f"### Nemotron {model}")
            st.markdown(f"**Original** {LIGHTS[o.light][0]} {LIGHTS[o.light][1]}  \n"
                        f"First-listed candidate wins **{o.first_rate:.0%}**")
            st.markdown(f"**Rewritten** {LIGHTS[f.light][0]} {LIGHTS[f.light][1]}  \n"
                        f"First-listed candidate wins **{f.first_rate:.0%}**")
            st.markdown(f"**Recommendation:** {ADVICE[choice]}")

    # --- chart: first-wins rate, original vs rewritten, 50% = fair
    chart_df = df[["Model", "Prompt", "first_rate", "p", "n"]].copy()
    chart_df["First-listed wins"] = chart_df["first_rate"]
    bars = alt.Chart(chart_df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=28).encode(
        x=alt.X("Model:N", title=None, sort=result["models"], axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("Prompt:N", sort=["Original", "Rewritten"]),
        y=alt.Y("First-listed wins:Q", title="How often the first-listed candidate wins",
                scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", tickCount=5)),
        color=alt.Color("Prompt:N", sort=["Original", "Rewritten"],
                        scale=alt.Scale(domain=["Original", "Rewritten"],
                                        range=[COLOR_ORIGINAL, COLOR_FIXED]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("Model:N"), alt.Tooltip("Prompt:N"),
                 alt.Tooltip("First-listed wins:Q", format=".0%"),
                 alt.Tooltip("p:Q", title="p-value", format=".3f"),
                 alt.Tooltip("n:Q", title="Decisions")],
    )
    fair_line = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(strokeDash=[4, 4], strokeWidth=2).encode(y="y:Q")
    fair_label = alt.Chart(pd.DataFrame({"y": [0.5], "t": ["50% = fair"]})).mark_text(
        align="left", dx=4, dy=-8, fontSize=12).encode(y="y:Q", text="t:N", x=alt.value(0))
    st.altair_chart((bars + fair_line + fair_label).properties(height=320), width="stretch")

    # --- detail table
    with st.expander("Detailed statistics"):
        show = pd.DataFrame({
            "Model": df.Model, "Prompt": df.Prompt,
            "First-listed wins": df.first_rate.map("{:.0%}".format),
            "Order bias": df.bias.map("{:+.0%}".format),
            "p-value": df.p.map("{:.3f}".format),
            "Picks clear winner": df.clear_correct.map("{:.0%}".format),
            "Order flips (swap test)": df.human.map("{:.0%}".format),
            "Decisions": df.n, "Unreadable": df.unreadable,
        })
        st.dataframe(show, hide_index=True, width="stretch")
        st.caption("Order bias = 2 × (first-listed wins − 50%). p-value tests whether the "
                   "first-listed win rate differs from 50%. Order flips = share of matched "
                   "trials where swapping the candidates changed the pick.")

    # --- the fixed prompt
    st.subheader("4. Your rewritten prompt")
    st.caption(f"Written by {result['author']} and re-tested above.")
    st.code(result["fixed"], language=None)
    left, right = st.columns(2)
    left.download_button("Download rewritten prompt", result["fixed"],
                         file_name="second_thought_prompt.txt")
    right.download_button("Download raw trial data (CSV)",
                          pd.DataFrame(rows).to_csv(index=False),
                          file_name="second_thought_trials.csv")

    with st.expander("Method and limitations"):
        st.markdown(
            "- **Design:** each matched pair is shown in both orders; a fair prompt picks the "
            "first-listed candidate 50% of the time.\n"
            "- **Judgement check:** a clearly stronger candidate must still win, so a prompt "
            "that guesses at random cannot pass.\n"
            "- **Scope:** this version tests order bias only. Framing, anchoring and demographic "
            "bias are not covered yet.\n"
            "- **Test set:** synthetic CVs. Re-run with your own anonymised data before deployment.\n"
            "- A low bias score means *less biased on these tests*, not *fair in every sense*.")
