import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

st.set_page_config(page_title="RPL 6 Experience Centre", layout="wide")

# =====================
# Theme C: Lovable-dark + subtle blue accents
# =====================
st.markdown(
    """
<style>
html, body, [data-testid="stAppViewContainer"] { background: #0b0f14 !important; color: rgba(255,255,255,0.92) !important; }
section[data-testid="stSidebar"] > div { background: #0b0f14 !important; border-right: 1px solid rgba(255,255,255,0.06); }
h1, h2, h3, h4 { color: rgba(255,255,255,0.96) !important; }
a, a:visited { color: #2d7ff9 !important; }
.rpl-banner { border-radius: 20px; padding: 18px 18px 14px 18px; background: linear-gradient(90deg, rgba(45,127,249,0.16), rgba(255,255,255,0.04)); border: 1px solid rgba(255,255,255,0.08); box-shadow: 0 12px 28px rgba(0,0,0,0.28); margin-bottom: 14px; }
.rpl-card { border-radius: 18px; padding: 14px 16px; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); box-shadow: 0 10px 24px rgba(0,0,0,0.22); }
.rpl-pill { display: inline-block; padding: 4px 10px; border-radius: 999px; background: rgba(45,127,249,0.18); border: 1px solid rgba(45,127,249,0.25); color: rgba(255,255,255,0.92); font-size: 12px; }
.rpl-muted { opacity: 0.78; font-size: 13px; }
.block-container {padding-top: 1.0rem; padding-bottom: 2rem;}
</style>
""",
    unsafe_allow_html=True,
)

# =====================
# Helpers
# =====================
def clean_player(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip())

def is_blank(x):
    return pd.isna(x) or (isinstance(x, str) and str(x).strip() == "")

def load_master(uploaded_file) -> pd.DataFrame:
    # Make failures visible in the UI (instead of Streamlit "Oh no")
    try:
        xls = pd.ExcelFile(uploaded_file)
        sheet = "Summary" if "Summary" in xls.sheet_names else xls.sheet_names[0]
        return pd.read_excel(uploaded_file, sheet_name=sheet, header=None)
    except Exception as e:
        st.error(f"Failed to read the uploaded Excel. Error: {e}")
        st.stop()

def parse_drop_groups(raw: pd.DataFrame):
    row0 = raw.iloc[0].astype(str)
    starts = [(i, row0[i]) for i in range(len(row0)) if isinstance(row0[i], str) and row0[i].startswith("Drop")]
    if not starts:
        st.error("Could not find any 'Drop X' headers in row 1 of the sheet. Please confirm the template.")
        st.stop()

    groups = []
    for idx, (start, label) in enumerate(starts):
        end = starts[idx+1][0] if idx+1 < len(starts) else raw.shape[1]
        width = end - start
        m = re.search(r"Drop\s+(\d+)", label)
        drop_num = int(m.group(1)) if m else None
        response_col = start
        question_text = raw.iloc[2, response_col] if raw.shape[0] > 2 else f"Q{drop_num}"
        if width == 2:
            pp_col = None
            bucket_col = start + 1
        else:
            pp_col = start + 1
            bucket_col = start + 2
        groups.append({
            "drop_num": drop_num,
            "response_col": response_col,
            "pp_col": pp_col,
            "bucket_col": bucket_col,
            "question_text": str(question_text).strip() if pd.notna(question_text) else f"Q{drop_num}",
        })
    return groups

def compute_streaks(answered_flags):
    longest = 0
    curr = 0
    for a in answered_flags:
        if a:
            curr += 1
            longest = max(longest, curr)
        else:
            curr = 0
    current = 0
    for a in reversed(answered_flags):
        if a:
            current += 1
        else:
            break
    return longest, current

def volatility_from_buckets(buckets):
    buckets = [b for b in buckets if b in ["H", "M", "L"]]
    if len(buckets) <= 1:
        return 0.0
    changes = sum(1 for i in range(1, len(buckets)) if buckets[i] != buckets[i-1])
    return (changes / (len(buckets) - 1)) * 100.0

def avg_risk_from_buckets(buckets):
    mp = {"H": 0.0, "M": 0.5, "L": 1.0}
    vals = [mp.get(b, np.nan) for b in buckets]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else 0.0

ARCH_DEF = {
    "Strategist": "Plays the long game: steady participation with selective conviction.",
    "Anchor": "Stays steady: consistent style across drops with low swing.",
    "Maverick": "Backs instinct: frequently comfortable against the room.",
    "Wildcard": "High-variance: bold calls + unpredictability create big upside.",
    "Calibrator": "Adapts actively: shifts style as signals evolve.",
    "Wiseman": "Rides clarity: leans into strong consensus.",
    "Pragmatist": "Situational: mixes styles without strong extremes.",
    "Ghost": "Not enough signal yet to infer a stable style."
}

# =====================
# Header
# =====================
st.markdown(
    """
<div class="rpl-banner">
  <div style="font-size: 26px; font-weight: 800; letter-spacing: 0.2px;">RPL 6 Experience Centre</div>
  <div style="opacity: 0.90; margin-top: 4px; font-size: 14px;">
    A behavioral analysis of each player's predictions and patterns
    <span class="rpl-pill" style="margin-left:10px;">Last updated: Drop 25</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Inputs")
    master_file = st.file_uploader("Upload Master Data (Drop 25)", type=["xlsx"])
    one_liners_file = st.file_uploader("Optional: Upload edited one-liners Excel", type=["xlsx"])
    show_debug = st.toggle("Show debug", value=False)

if master_file is None:
    st.info("Upload the Drop 25 master Excel to render the dashboard.")
    st.stop()

raw = load_master(master_file)
raw = raw.replace({"\u00a0": " "}, regex=True)

groups = parse_drop_groups(raw)
valid_groups = [g for g in groups if g["drop_num"] not in [None, 12]]
valid_groups = sorted(valid_groups, key=lambda x: x["drop_num"])
VALID_DROPS = [g["drop_num"] for g in valid_groups]
TOTAL_VALID_DROPS = len(VALID_DROPS)

# Player rows: names in column C (index 2), rows 4..51 => 3..50
player_rows = list(range(3, 51))
player_to_row = {}
players_in_file = []
for r in player_rows:
    nm = raw.iloc[r, 2] if raw.shape[1] > 2 else None
    if is_blank(nm):
        continue
    c = clean_player(nm)
    players_in_file.append(c)
    player_to_row[c] = r

if len(players_in_file) == 0:
    st.error("No players found in column C rows 4–51. Please confirm the template.")
    st.stop()

q_lookup = {g["drop_num"]: g["question_text"] for g in valid_groups}

records = []
for name in players_in_file:
    r = player_to_row[name]
    answered_flags = []
    buckets_seq = []
    H = M = L = 0
    pp_yes = []
    pp_buckets = {}

    for g in valid_groups:
        resp = raw.iloc[r, g["response_col"]]
        answered = not is_blank(resp)
        answered_flags.append(answered)

        b = raw.iloc[r, g["bucket_col"]] if g["bucket_col"] is not None else np.nan
        b = b.strip() if isinstance(b, str) else ""
        if answered and b in ["H", "M", "L"]:
            buckets_seq.append(b)
            if b == "H": H += 1
            elif b == "M": M += 1
            elif b == "L": L += 1

        if g["pp_col"] is not None:
            pp = raw.iloc[r, g["pp_col"]]
            if isinstance(pp, str) and pp.strip().lower() == "yes":
                pp_yes.append(g["drop_num"])
                if answered and b in ["H", "M", "L"]:
                    pp_buckets[g["drop_num"]] = b

    attended = int(sum(answered_flags))
    total_bucketed = H + M + L
    Hpct = (H / total_bucketed * 100.0) if total_bucketed else 0.0
    Mpct = (M / total_bucketed * 100.0) if total_bucketed else 0.0
    Lpct = (L / total_bucketed * 100.0) if total_bucketed else 0.0
    vol = volatility_from_buckets(buckets_seq)
    risk = avg_risk_from_buckets(buckets_seq)
    longest, current = compute_streaks(answered_flags)

    pp_used_total = len(pp_yes)
    pp_valid = pp_yes[:2]
    pp_q1 = pp_valid[0] if len(pp_valid) >= 1 else np.nan
    pp_q2 = pp_valid[1] if len(pp_valid) >= 2 else np.nan
    pp_b1 = pp_buckets.get(pp_q1, "") if pp_q1 == pp_q1 else ""
    pp_b2 = pp_buckets.get(pp_q2, "") if pp_q2 == pp_q2 else ""

    def bucket_label(b):
        return {"H": "crowd pick", "M": "balanced pick", "L": "contrarian pick"}.get(b, "")

    pp_lines = []
    if pp_q1 == pp_q1:
        pp_lines.append(f"**First Power Play:** Q{int(pp_q1)} — {q_lookup.get(int(pp_q1), '')}  \n*({bucket_label(pp_b1)})*")
    if pp_q2 == pp_q2:
        pp_lines.append(f"**Second Power Play:** Q{int(pp_q2)} — {q_lookup.get(int(pp_q2), '')}  \n*({bucket_label(pp_b2)})*")
    if pp_used_total > 2:
        pp_lines.append(f"*You marked Power Play {pp_used_total} times — only the first two count.*")
    pp_moments = "\n\n".join(pp_lines) if pp_lines else "No Power Play moments recorded yet."

    records.append({
        "Player_Clean": name,
        "Attended": attended,
        "TotalDrops": TOTAL_VALID_DROPS,
        "AttendancePct": (attended / TOTAL_VALID_DROPS * 100.0) if TOTAL_VALID_DROPS else 0.0,
        "H%": Hpct, "M%": Mpct, "L%": Lpct,
        "Volatility%": vol, "AvgRisk": risk,
        "LongestStreak": longest, "CurrentStreak": current,
        "PP_used_total": pp_used_total,
        "PP_Q1": pp_q1, "PP_Q2": pp_q2,
        "PP_b1": pp_b1, "PP_b2": pp_b2,
        "PP_moments": pp_moments,
    })

df = pd.DataFrame(records)
df["AttendancePct_Calc"] = (df["Attended"] / df["TotalDrops"]) * 100.0

# Archetype assignment based on quantiles
base = df[df["Attended"] > 0].copy()
H75 = float(base["H%"].quantile(0.75)) if len(base) else 50.0
L75 = float(base["L%"].quantile(0.75)) if len(base) else 15.0
V75 = float(base["Volatility%"].quantile(0.75)) if len(base) else 60.0
V25 = float(base["Volatility%"].quantile(0.25)) if len(base) else 30.0
R75 = float(base["AvgRisk"].quantile(0.75)) if len(base) else 0.6

def assign_arch(r):
    if r["AttendancePct"] < 25:
        return "Ghost"
    if r["AvgRisk"] >= R75 and r["Volatility%"] >= V25:
        return "Wildcard"
    if r["L%"] >= L75 and r["H%"] <= 60:
        return "Maverick"
    if r["H%"] >= H75 and r["L%"] <= 10:
        return "Wiseman"
    if r["Volatility%"] <= V25 and r["AttendancePct"] >= 50:
        return "Anchor"
    if r["Volatility%"] >= V75:
        return "Calibrator"
    if r["AttendancePct"] >= 75 and r["PP_used_total"] <= 1:
        return "Strategist"
    return "Pragmatist"

df["Archetype"] = df.apply(assign_arch, axis=1)
df["ArchetypeDef"] = df["Archetype"].map(ARCH_DEF).fillna("")

# Merge one-liners if provided
df["OneLiner"] = np.nan
if one_liners_file is not None:
    try:
        ol = pd.read_excel(one_liners_file)
        if "Player" in ol.columns:
            ol["Player_Clean"] = ol["Player"].astype(str).apply(clean_player)
            text_col = None
            for c in ol.columns:
                if "one-liner" in str(c).lower() or "unique" in str(c).lower():
                    text_col = c
                    break
            if text_col is not None:
                df = df.merge(ol[["Player_Clean", text_col]].rename(columns={text_col: "OneLiner"}),
                              on="Player_Clean", how="left")
    except Exception:
        pass

df["OneLiner"] = df["OneLiner"].fillna("Not much data to infer any prediction patterns.")

# =====================
# Minimal render to avoid crash; expand after app runs
# =====================
st.markdown("## Quick check")
st.write(f"Loaded **{len(df)}** players and **{TOTAL_VALID_DROPS}** valid drops.")
st.dataframe(df.head(10), hide_index=True, use_container_width=True)

if show_debug:
    st.write("Valid drops:", VALID_DROPS)
    st.write("Example question lookup:", list(q_lookup.items())[:3])
