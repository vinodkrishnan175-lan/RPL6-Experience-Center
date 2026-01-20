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
html, body, [data-testid="stAppViewContainer"] {
  background: #0b0f14 !important;
  color: rgba(255,255,255,0.92) !important;
}
section[data-testid="stSidebar"] > div {
  background: #0b0f14 !important;
  border-right: 1px solid rgba(255,255,255,0.06);
}
h1, h2, h3, h4 { color: rgba(255,255,255,0.96) !important; }
a, a:visited { color: #2d7ff9 !important; }

.rpl-banner {
  border-radius: 20px;
  padding: 18px 18px 14px 18px;
  background: linear-gradient(90deg, rgba(45,127,249,0.16), rgba(255,255,255,0.04));
  border: 1px solid rgba(255,255,255,0.08);
  box-shadow: 0 12px 28px rgba(0,0,0,0.28);
  margin-bottom: 14px;
}
.rpl-card {
  border-radius: 18px;
  padding: 14px 16px;
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  box-shadow: 0 10px 24px rgba(0,0,0,0.22);
}
.rpl-pill {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  background: rgba(45,127,249,0.18);
  border: 1px solid rgba(45,127,249,0.25);
  color: rgba(255,255,255,0.92);
  font-size: 12px;
}
.rpl-muted { opacity: 0.78; font-size: 13px; }
.rpl-yellow { color: rgba(255, 209, 102, 0.95); }

/* ? help button */
button[kind="secondary"] {
  border-radius: 999px !important;
  padding: 0.05rem 0.45rem !important;
  min-height: 0 !important;
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  color: rgba(255,255,255,0.90) !important;
}

/* Tighter container padding */
.block-container {padding-top: 1.0rem; padding-bottom: 2rem;}
</style>
""",
    unsafe_allow_html=True,
)

# =====================
# Data helpers
# =====================
def load_master(uploaded_file) -> pd.DataFrame:
    return pd.read_excel(uploaded_file, sheet_name="Summary", header=None)

def clean_player(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip())

def parse_drop_groups(raw: pd.DataFrame):
    """Return list of dicts for each drop: {drop_label, drop_num, start_col, width, response_col, pp_col, bucket_col, question_text}"""
    row0 = raw.iloc[0].astype(str)
    starts = [(i, row0[i]) for i in range(len(row0)) if isinstance(row0[i], str) and row0[i].startswith("Drop")]
    groups = []
    for idx, (start, label) in enumerate(starts):
        end = starts[idx+1][0] if idx+1 < len(starts) else raw.shape[1]
        width = end - start
        # drop number
        m = re.search(r"Drop\s+(\d+)", label)
        drop_num = int(m.group(1)) if m else None
        # question text is in row2 at response_col
        response_col = start
        question_text = raw.iloc[2, response_col]
        # identify columns based on width (2 or 3)
        if width == 2:
            pp_col = None
            bucket_col = start + 1
        else:
            pp_col = start + 1
            bucket_col = start + 2
        groups.append({
            "drop_label": label,
            "drop_num": drop_num,
            "start_col": start,
            "width": width,
            "response_col": response_col,
            "pp_col": pp_col,
            "bucket_col": bucket_col,
            "question_text": str(question_text).strip() if pd.notna(question_text) else f"Q{drop_num}"
        })
    return groups

def is_blank(x):
    return pd.isna(x) or (isinstance(x, str) and str(x).strip() == "")

def compute_streaks(answered_flags):
    # longest streak of 1s
    longest = 0
    curr = 0
    for a in answered_flags:
        if a:
            curr += 1
            longest = max(longest, curr)
        else:
            curr = 0
    # current streak ending at last element
    current = 0
    for a in reversed(answered_flags):
        if a:
            current += 1
        else:
            break
    return longest, current

def volatility_from_buckets(buckets):
    # buckets list includes only answered drops buckets (H/M/L)
    buckets = [b for b in buckets if isinstance(b, str) and b.strip() in ["H","M","L"]]
    if len(buckets) <= 1:
        return 0.0
    changes = sum(1 for i in range(1, len(buckets)) if buckets[i] != buckets[i-1])
    return (changes / (len(buckets)-1)) * 100.0

def avg_risk_from_buckets(buckets):
    mp = {"H": 0.0, "M": 0.5, "L": 1.0}
    vals = [mp.get(str(b).strip(), np.nan) for b in buckets]
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
# UI: Uploads
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
    st.caption("Tip: If you upload one-liners, the player narrative will use your edited copy.")
    show_debug = st.toggle("Show debug", value=False)

if master_file is None:
    st.info("Upload the Drop 25 master Excel to render the dashboard.")
    st.stop()

raw = load_master(master_file)
raw = raw.replace({"\u00a0": " "}, regex=True)  # nbsp

# Parse drop groups and exclude scrapped drop 12
groups = parse_drop_groups(raw)
valid_groups = [g for g in groups if g["drop_num"] != 12 and g["drop_num"] is not None]
valid_groups = sorted(valid_groups, key=lambda x: x["drop_num"])

VALID_DROPS = [g["drop_num"] for g in valid_groups]
TOTAL_VALID_DROPS = len(VALID_DROPS)  # should be 24 when 25 drops with 12 scrapped

# Player rows are 3..50 (based on your format: names start at row index 3)
player_rows = list(range(3, 51))
players = []
for r in player_rows:
    name = raw.iloc[r, 2]
    if is_blank(name):
        continue
    players.append(clean_player(name))

# Build per-player metrics from raw
records = []
for r, name in zip(player_rows, players):
    answered_flags = []
    buckets_seq = []
    H = M = L = 0
    pp_yes = []  # list of drop nums where PP marked yes (chronological)
    pp_buckets = {}
    for g in valid_groups:
        resp = raw.iloc[r, g["response_col"]]
        answered = not is_blank(resp)
        answered_flags.append(answered)

        # bucket
        b = raw.iloc[r, g["bucket_col"]] if g["bucket_col"] is not None else np.nan
        if answered and isinstance(b, str):
            b = b.strip()
        if answered and b in ["H","M","L"]:
            buckets_seq.append(b)
            if b == "H": H += 1
            elif b == "M": M += 1
            elif b == "L": L += 1

        # PP
        if g["pp_col"] is not None:
            pp = raw.iloc[r, g["pp_col"]]
            if isinstance(pp, str) and pp.strip().lower() == "yes":
                pp_yes.append(g["drop_num"])
                if answered and b in ["H","M","L"]:
                    pp_buckets[g["drop_num"]] = b

    attended = int(sum(answered_flags))
    total_bucketed = H + M + L
    Hpct = (H / total_bucketed * 100.0) if total_bucketed else 0.0
    Mpct = (M / total_bucketed * 100.0) if total_bucketed else 0.0
    Lpct = (L / total_bucketed * 100.0) if total_bucketed else 0.0
    vol = volatility_from_buckets(buckets_seq)
    risk = avg_risk_from_buckets(buckets_seq)
    longest_streak, current_streak = compute_streaks(answered_flags)

    # PP valid (first 2)
    pp_used_total = len(pp_yes)
    pp_valid = pp_yes[:2]
    pp_q1 = pp_valid[0] if len(pp_valid) >= 1 else np.nan
    pp_q2 = pp_valid[1] if len(pp_valid) >= 2 else np.nan
    pp_b1 = pp_buckets.get(pp_q1, "") if pp_q1==pp_q1 else ""
    pp_b2 = pp_buckets.get(pp_q2, "") if pp_q2==pp_q2 else ""

    # Question lookup for PP
    q_lookup = {g["drop_num"]: g["question_text"] for g in valid_groups}
    pp_t1 = q_lookup.get(pp_q1, "") if pp_q1==pp_q1 else ""
    pp_t2 = q_lookup.get(pp_q2, "") if pp_q2==pp_q2 else ""

    # Build PP readable text
    def bucket_label(b):
        return {"H": "crowd pick", "M": "balanced pick", "L": "contrarian pick"}.get(b, "")
    pp_lines = []
    if pp_q1==pp_q1:
        pp_lines.append(f"**First Power Play:** Q{pp_q1} — {pp_t1}  \n*({bucket_label(pp_b1)})*")
    if pp_q2==pp_q2:
        pp_lines.append(f"**Second Power Play:** Q{pp_q2} — {pp_t2}  \n*({bucket_label(pp_b2)})*")
    if pp_used_total > 2:
        pp_lines.append(f"*You marked Power Play {pp_used_total} times — only the first two count.*")
    pp_moments = "\n\n".join(pp_lines) if pp_lines else "No Power Play moments recorded yet."

    records.append({
        "Player": name,
        "Player_Clean": clean_player(name),
        "Attended": attended,
        "TotalDrops": TOTAL_VALID_DROPS,
        "AttendancePct": (attended / TOTAL_VALID_DROPS * 100.0) if TOTAL_VALID_DROPS else 0.0,
        "H%": Hpct,
        "M%": Mpct,
        "L%": Lpct,
        "Volatility%": vol,
        "AvgRisk": risk,
        "LongestStreak": longest_streak,
        "CurrentStreak": current_streak,
        "PP_used_total": pp_used_total,
        "PP_Q1": pp_q1,
        "PP_Q2": pp_q2,
        "PP_b1": pp_b1,
        "PP_b2": pp_b2,
        "PP_moments": pp_moments,
    })

df = pd.DataFrame(records)

# Attendance calc for plotting
df["AttendancePct_Calc"] = (df["Attended"] / df["TotalDrops"]) * 100.0

# ---------------------
# Archetype assignment (simple + threshold shown)
# ---------------------
# Use quantiles among players with some participation
base = df[df["Attended"] > 0].copy()
H75 = base["H%"].quantile(0.75) if len(base) else 50
L75 = base["L%"].quantile(0.75) if len(base) else 15
V75 = base["Volatility%"].quantile(0.75) if len(base) else 60
V25 = base["Volatility%"].quantile(0.25) if len(base) else 30
R75 = base["AvgRisk"].quantile(0.75) if len(base) else 0.6

def assign_arch(r):
    att_pct = r["AttendancePct"]
    if att_pct < 25:
        return "Ghost"
    # Wildcard first (high risk + some volatility)
    if r["AvgRisk"] >= R75 and r["Volatility%"] >= (V25):
        return "Wildcard"
    if r["L%"] >= L75 and r["H%"] <= 60:
        return "Maverick"
    if r["H%"] >= H75 and r["L%"] <= 10:
        return "Wiseman"
    if r["Volatility%"] <= V25 and att_pct >= 50:
        return "Anchor"
    if r["Volatility%"] >= V75:
        return "Calibrator"
    if att_pct >= 75 and r["PP_used_total"] <= 1:
        return "Strategist"
    return "Pragmatist"

df["Archetype"] = df.apply(assign_arch, axis=1)
df["ArchetypeDef"] = df["Archetype"].map(ARCH_DEF).fillna("")

# ---------------------
# One-liners: merge if provided
# ---------------------
if one_liners_file is not None:
    try:
        ol = pd.read_excel(one_liners_file)
        # expected columns: Player, Unique one-liner (editable)
        ol_cols = {c.lower(): c for c in ol.columns}
        pcol = ol_cols.get("player", None)
        tcol = None
        for k in ol_cols:
            if "one-liner" in k or "unique" in k:
                tcol = ol_cols[k]
                break
        if pcol and tcol:
            ol["Player_Clean"] = ol[pcol].astype(str).apply(clean_player)
            ol2 = ol[["Player_Clean", tcol]].rename(columns={tcol: "OneLiner"})
            df = df.merge(ol2, on="Player_Clean", how="left")
        else:
            df["OneLiner"] = np.nan
    except Exception:
        df["OneLiner"] = np.nan
else:
    df["OneLiner"] = np.nan

def fallback_oneliner(r):
    if r["Attended"] == 0:
        return "Not enough data to infer any prediction patterns."
    style_bits = []
    if r["L%"] >= L75: style_bits.append("leans against the crowd")
    if r["H%"] >= H75: style_bits.append("rides consensus when it’s strong")
    if r["Volatility%"] >= V75: style_bits.append("changes gears often")
    if r["Volatility%"] <= V25: style_bits.append("stays remarkably steady")
    if not style_bits: style_bits.append("mixes styles depending on the question")
    return f"{r['Player_Clean']}: {style_bits[0].capitalize()} — that could be a real edge on February reveal days."

df["OneLiner"] = df["OneLiner"].fillna(df.apply(fallback_oneliner, axis=1))

# =====================
# SECTION 1 — The Room
# =====================
st.markdown("## Section 1 — The Room")

# Room-level aggregates
total_players = df["Player_Clean"].nunique()
active_25 = int((df["AttendancePct"] >= 25).sum())
total_responses = int(df["Attended"].sum())
# H/M/L mix overall (use counts derived from % * attended approximations is risky); compute using buckets from df % and attended counts approximated:
# Better: compute overall H/M/L from raw buckets directly
H_total = M_total = L_total = 0
for r_i, name in zip(player_rows, players):
    for g in valid_groups:
        resp = raw.iloc[r_i, g["response_col"]]
        if is_blank(resp): 
            continue
        b = raw.iloc[r_i, g["bucket_col"]]
        if isinstance(b, str):
            b = b.strip()
        if b == "H": H_total += 1
        elif b == "M": M_total += 1
        elif b == "L": L_total += 1
total_bucketed = H_total + M_total + L_total
Hmix = (H_total/total_bucketed*100) if total_bucketed else 0
Mmix = (M_total/total_bucketed*100) if total_bucketed else 0
Lmix = (L_total/total_bucketed*100) if total_bucketed else 0

pp_exhausted = int((df["PP_used_total"].clip(upper=2) == 2).sum())
pp_none = int((df["PP_used_total"] == 0).sum())
pp_once = int((df["PP_used_total"] == 1).sum())

perfect_attendance = df[df["Attended"] == TOTAL_VALID_DROPS].sort_values("Player_Clean")
streak10 = df[df["CurrentStreak"] >= 10].sort_values(["CurrentStreak","Player_Clean"], ascending=[False,True])

# KPI tiles
k1, k2, k3, k4 = st.columns(4, gap="large")
with k1:
    st.markdown("<div class='rpl-card'>", unsafe_allow_html=True)
    st.markdown(f"### {total_players} <span class='rpl-muted'>players</span>", unsafe_allow_html=True)
    st.markdown("<div class='rpl-muted'>Total participants</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with k2:
    st.markdown("<div class='rpl-card'>", unsafe_allow_html=True)
    st.markdown(f"### {active_25} <span class='rpl-muted'>active</span>", unsafe_allow_html=True)
    st.markdown("<div class='rpl-muted'>≥25% attendance</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with k3:
    st.markdown("<div class='rpl-card'>", unsafe_allow_html=True)
    st.markdown(f"### {TOTAL_VALID_DROPS} <span class='rpl-muted'>drops</span>", unsafe_allow_html=True)
    st.markdown("<div class='rpl-muted'>Valid drops so far (Drop 12 excluded)</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with k4:
    st.markdown("<div class='rpl-card'>", unsafe_allow_html=True)
    st.markdown(f"### {total_responses} <span class='rpl-muted'>responses</span>", unsafe_allow_html=True)
    st.markdown("<div class='rpl-muted'>Total answers submitted</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

k5, k6, k7, k8 = st.columns(4, gap="large")
with k5:
    st.markdown("<div class='rpl-card'>", unsafe_allow_html=True)
    st.markdown("### H / M / L mix")
    fig_mix = px.pie(names=["H","M","L"], values=[H_total, M_total, L_total], hole=0.6)
    fig_mix.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=220, showlegend=True)
    st.plotly_chart(fig_mix, use_container_width=True)
    st.markdown(f"<div class='rpl-muted'>H {Hmix:.1f}% · M {Mmix:.1f}% · L {Lmix:.1f}%</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with k6:
    st.markdown("<div class='rpl-card'>", unsafe_allow_html=True)
    st.markdown(f"### {pp_exhausted} <span class='rpl-muted'>exhausted</span>", unsafe_allow_html=True)
    st.markdown("<div class='rpl-muted'>Players who used both valid Power Plays</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with k7:
    st.markdown("<div class='rpl-card'>", unsafe_allow_html=True)
    st.markdown(f"### {len(streak10)} <span class='rpl-muted'>streak 10+</span>", unsafe_allow_html=True)
    st.markdown("<div class='rpl-muted'>Players currently on 10+ no-miss streak</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with k8:
    st.markdown("<div class='rpl-card'>", unsafe_allow_html=True)
    st.markdown(f"### {len(perfect_attendance)} <span class='rpl-muted'>perfect</span>", unsafe_allow_html=True)
    st.markdown("<div class='rpl-muted'>Answered every valid drop so far</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Metric explainer + archetypes
st.markdown("<div class='rpl-card'>", unsafe_allow_html=True)
st.markdown("### How to read the metrics")
cA, cB = st.columns([1.2, 1.8], gap="large")
with cA:
    st.markdown("- **H%**: picked the most popular option (crowd-aligned)")
    st.markdown("- **M%**: middle-of-the-road picks")
    st.markdown("- **L%**: least popular option (contrarian)")
    st.markdown("- **Volatility**: how often your H/M/L bucket changes across answered drops")
    st.markdown("- **AvgRisk**: H=0, M=0.5, L=1 averaged across answered drops")
with cB:
    st.markdown("### Archetypes (simple meaning + thresholds)")
    st.markdown(f"- **Wiseman**: high H% (≥{H75:.1f}) and very low L% (≤10)")
    st.markdown(f"- **Maverick**: high L% (≥{L75:.1f}) with H% not dominant (≤60)")
    st.markdown(f"- **Wildcard**: high AvgRisk (≥{R75:.2f}) and at least moderate volatility (≥{V25:.1f})")
    st.markdown(f"- **Anchor**: very low volatility (≤{V25:.1f}) with solid attendance (≥50%)")
    st.markdown(f"- **Calibrator**: high volatility (≥{V75:.1f})")
    st.markdown(f"- **Strategist**: high attendance (≥75%) + conservative PP usage (≤1)")
    st.markdown(f"- **Pragmatist**: everything else (balanced mix)")
    st.markdown(f"- **Ghost**: attendance < 25%")
st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# =====================
# SECTION 2 — Player Experience Centre
# =====================
st.markdown("## Section 2 — Player Experience Centre")

players_sorted = sorted(df["Player_Clean"].tolist())
sel = st.selectbox("Select player", players_sorted, index=0)
row = df[df["Player_Clean"] == sel].iloc[0]

col1, col2, col3 = st.columns([1.25, 1.05, 1.25], gap="large")

# Column 1: Identity + narrative
with col1:
    st.markdown("<div class='rpl-card'>", unsafe_allow_html=True)
    st.markdown(f"### {row['Player_Clean']}")
    st.markdown(f"<span class='rpl-pill'>{row['Archetype']}</span>", unsafe_allow_html=True)
    st.markdown(f"<div class='rpl-muted' style='margin-top:8px;'>{row['ArchetypeDef']}</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Attendance**")
    st.progress(row["Attended"]/row["TotalDrops"] if row["TotalDrops"] else 0.0, text=f"{int(row['Attended'])} / {int(row['TotalDrops'])} drops")
    pp_used_capped = min(int(row["PP_used_total"]), 2)
    st.markdown("**Power Plays used**")
    st.progress(pp_used_capped/2, text=f"{pp_used_capped} / 2 (only first 2 count)")
    st.markdown("---")
    st.markdown("### What’s unique about you")
    st.write(row["OneLiner"])
    st.markdown("</div>", unsafe_allow_html=True)

# Column 2: Metrics + visuals
with col2:
    st.markdown("<div class='rpl-card'>", unsafe_allow_html=True)
    st.markdown("### Your Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("H%", f"{row['H%']:.1f}%")
    m2.metric("M%", f"{row['M%']:.1f}%")
    m3.metric("L%", f"{row['L%']:.1f}%")
    s1, s2 = st.columns(2)
    s1.metric("Volatility", f"{row['Volatility%']:.1f}%")
    s2.metric("AvgRisk", f"{row['AvgRisk']:.2f}")
    t1, t2 = st.columns(2)
    t1.metric("Current streak", f"{int(row['CurrentStreak'])}")
    t2.metric("Longest streak", f"{int(row['LongestStreak'])}")
    st.markdown("---")
    fig_player = px.pie(names=["H","M","L"], values=[row["H%"], row["M%"], row["L%"]], hole=0.6)
    fig_player.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=240, showlegend=True)
    st.plotly_chart(fig_player, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Column 3: Power Play + conviction signature + timeline
with col3:
    st.markdown("<div class='rpl-card'>", unsafe_allow_html=True)
    st.markdown("### Power Play Moments")
    st.markdown(row["PP_moments"])
    # Conviction Signature
    def conviction_signature(pp_used_total, b1, b2):
        if pp_used_total == 0:
            return "Conviction held back — you still have both swing levers for February."
        buckets = [b for b in [b1, b2] if isinstance(b, str) and b]
        if "L" in buckets:
            return "Instinct-driven conviction — you’re willing to bet big away from consensus."
        if "H" in buckets:
            return "Crowd-leverage conviction — you amplify moments where the room is collectively sure."
        if "M" in buckets:
            return "Measured conviction — you deploy Power Plays in stable lanes."
        return "Conviction layered in — reveal days will amplify what you backed."
    st.markdown("---")
    st.markdown("### Conviction Signature")
    st.write(conviction_signature(int(row["PP_used_total"]), str(row["PP_b1"]), str(row["PP_b2"])))

    # Timeline of buckets (visual)
    # Build per-drop bucket sequence for this player using raw again
    p_row_idx = players_sorted.index(sel) + 3  # player starts at row 3 in raw
    seq = []
    for g in valid_groups:
        resp = raw.iloc[p_row_idx, g["response_col"]]
        if is_blank(resp):
            seq.append("—")
        else:
            b = raw.iloc[p_row_idx, g["bucket_col"]]
            b = b.strip() if isinstance(b, str) else "—"
            seq.append(b if b in ["H","M","L"] else "—")
    tl = pd.DataFrame({"Drop": [f"Q{d}" for d in VALID_DROPS], "Bucket": seq})
    # Map to numeric for plotting strip
    mp = {"H": 2, "M": 1, "L": 0, "—": np.nan}
    tl["Y"] = tl["Bucket"].map(mp)
    fig_tl = px.scatter(tl, x="Drop", y="Y", color="Bucket", hover_data={"Drop": True, "Bucket": True, "Y": False})
    fig_tl.update_layout(height=220, margin=dict(l=0,r=0,t=10,b=0), yaxis=dict(visible=False), xaxis=dict(tickangle=0))
    st.markdown("---")
    st.markdown("### Style timeline (H/M/L over questions)")
    st.plotly_chart(fig_tl, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# =====================
# SECTION 3 — Power Play Analytics + Story Mode
# =====================
st.markdown("## Section 3 — Power Plays + Story Mode")

# Power play status boxes
b1, b2, b3 = st.columns(3, gap="large")

exhausted = df[df["PP_used_total"].clip(upper=2) == 2].copy()
used_once = df[df["PP_used_total"] == 1].copy()
unused = df[df["PP_used_total"] == 0].copy()

with b1:
    st.markdown("<div class='rpl-card'>", unsafe_allow_html=True)
    st.markdown(f"### All-In Early ({len(exhausted)})")
    st.markdown("<div class='rpl-muted'>Used both valid Power Plays — shows when conviction was committed.</div>", unsafe_allow_html=True)
    show = exhausted.copy()
    show["Exhausted by"] = show["PP_Q2"].apply(lambda x: f"Q{int(x)}" if pd.notna(x) else "—")
    st.dataframe(show[["Player_Clean","Exhausted by"]].rename(columns={"Player_Clean":"Player"}).sort_values("Exhausted by"), hide_index=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with b2:
    st.markdown("<div class='rpl-card'>", unsafe_allow_html=True)
    st.markdown(f"### One Shot Used ({len(used_once)})")
    st.markdown("<div class='rpl-muted'>Only one Power Play used so far — one bullet still in reserve.</div>", unsafe_allow_html=True)
    show = used_once.copy()
    show["Used on"] = show["PP_Q1"].apply(lambda x: f"Q{int(x)}" if pd.notna(x) else "—")
    st.dataframe(show[["Player_Clean","Used on"]].rename(columns={"Player_Clean":"Player"}).sort_values("Used on"), hide_index=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with b3:
    st.markdown("<div class='rpl-card'>", unsafe_allow_html=True)
    st.markdown(f"### Conviction Still Loaded ({len(unused)})")
    st.markdown("<div class='rpl-muted'>No Power Plays used yet — maximum late-surge leverage.</div>", unsafe_allow_html=True)
    st.write(", ".join(sorted(unused["Player_Clean"].tolist())) if len(unused) else "—")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# Story mode carousel (tabs feel like swipe on mobile)
stories = st.tabs(["Contrarian League", "Most Volatile", "Wildcards", "Conviction Still Loaded", "All-In Early"])

# Apply attendance filter for story lists
story_base = df[df["AttendancePct"] >= 30].copy()

with stories[0]:
    st.markdown("<div class='rpl-card'>", unsafe_allow_html=True)
    st.markdown("### The Contrarian League")
    st.markdown("<div class='rpl-muted'>Most frequently picked the least popular option (L). High-upside if minority outcomes hit in February.</div>", unsafe_allow_html=True)
    top = story_base.sort_values("L%", ascending=False).head(10)
    st.dataframe(top[["Player_Clean","Archetype","L%"]].rename(columns={"Player_Clean":"Player"}), hide_index=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with stories[1]:
    st.markdown("<div class='rpl-card'>", unsafe_allow_html=True)
    st.markdown("### The Shape-Shifters")
    st.markdown("<div class='rpl-muted'>Highest volatility: switches between H/M/L most often. Adaptability can be an edge when conditions change.</div>", unsafe_allow_html=True)
    top = story_base.sort_values("Volatility%", ascending=False).head(10)
    st.dataframe(top[["Player_Clean","Archetype","Volatility%"]].rename(columns={"Player_Clean":"Player"}), hide_index=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with stories[2]:
    st.markdown("<div class='rpl-card'>", unsafe_allow_html=True)
    st.markdown("### The Wildcards")
    st.markdown("<div class='rpl-muted'>Different from contrarian: not just ‘least popular’ picks, but overall high-risk + high-variance style.</div>", unsafe_allow_html=True)
    top = story_base.sort_values("AvgRisk", ascending=False).head(10)
    st.dataframe(top[["Player_Clean","Archetype","AvgRisk","Volatility%"]].rename(columns={"Player_Clean":"Player"}), hide_index=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with stories[3]:
    st.markdown("<div class='rpl-card'>", unsafe_allow_html=True)
    st.markdown("### Conviction Still Loaded")
    st.markdown("<div class='rpl-muted'>No Power Plays used yet — two conviction chips still available for a February surge.</div>", unsafe_allow_html=True)
    show = story_base[story_base["PP_used_total"] == 0].sort_values(["AttendancePct","Player_Clean"], ascending=[False, True])
    st.write(", ".join(show["Player_Clean"].tolist()) if len(show) else "—")
    st.markdown("</div>", unsafe_allow_html=True)

with stories[4]:
    st.markdown("<div class='rpl-card'>", unsafe_allow_html=True)
    st.markdown("### All-In Early")
    st.markdown("<div class='rpl-muted'>Already used both valid Power Plays — now it’s about whether early conviction lands on reveal days.</div>", unsafe_allow_html=True)
    show = story_base[story_base["PP_used_total"].clip(upper=2) == 2].copy()
    show["Exhausted by"] = show["PP_Q2"].apply(lambda x: f"Q{int(x)}" if pd.notna(x) else "—")
    show = show.sort_values("Exhausted by").head(15)
    st.dataframe(show[["Player_Clean","Archetype","Exhausted by"]].rename(columns={"Player_Clean":"Player"}), hide_index=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# Risk vs Attendance map (optional visual for exploration)
st.markdown("### Risk vs Attendance Map")
fig = px.scatter(
    df,
    x="AttendancePct_Calc",
    y="AvgRisk",
    color="Archetype",
    hover_name="Player_Clean",
    hover_data={"Archetype": True, "AttendancePct_Calc": False, "AvgRisk": False},
    labels={"AttendancePct_Calc": "Attendance %", "AvgRisk": "AvgRisk"},
)
fig.update_layout(xaxis=dict(range=[0, 100]), yaxis=dict(range=[0, 1]), margin=dict(l=0,r=0,t=10,b=0), height=420)
st.plotly_chart(fig, use_container_width=True)

if show_debug:
    st.markdown("### Debug")
    st.write("Valid drops:", VALID_DROPS)
    st.write("Total valid drops:", TOTAL_VALID_DROPS)
    st.write("Perfect attendance names:", perfect_attendance["Player_Clean"].tolist())
    st.write("Current 10+ streak club:", streak10[["Player_Clean","CurrentStreak"]].to_dict("records"))
