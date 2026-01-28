import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re, difflib
from pathlib import Path

# ==============================
# RPL 6 Experience Centre — INTERIM (UI/UX + PP mapping fixes)
# Requires baked-in data files in repo:
#   data/drop25_metrics.xlsx
#   data/master_data.xlsx  (sheet: Summary)
# ==============================

# --- Page ---
st.set_page_config(page_title="RPL 6 Experience Centre", layout="wide")

# --- Styling ---
st.markdown(
    """
<style>
html, body, [data-testid="stAppViewContainer"] { background: #0b0f14 !important; color: rgba(255,255,255,0.92) !important; }
section[data-testid="stSidebar"] > div { background: #0b0f14 !important; border-right: 1px solid rgba(255,255,255,0.06); }
.block-container { padding-top: 1.0rem; padding-bottom: 2.0rem; max-width: 1200px; }
h1,h2,h3,h4 { color: rgba(255,255,255,0.96) !important; }

.rpl-banner {
  border-radius: 22px;
  padding: 32px 22px 26px 22px; /* slightly bigger ribbon */
  background: linear-gradient(90deg, rgba(45,127,249,0.30), rgba(6,84,130,0.14));
  border: 1px solid rgba(255,255,255,0.06);
  box-shadow: 0 18px 40px rgba(0,0,0,0.45);
  margin-bottom: 18px;
}

.rpl-card {
  border-radius: 16px;
  padding: 14px 16px;
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.06);
  box-shadow: 0 10px 22px rgba(0,0,0,0.22);
}

.rpl-pill {
  display: inline-block;
  padding: 6px 12px;
  border-radius: 999px;
  background: rgba(45,127,249,0.22);
  border: 1px solid rgba(45,127,249,0.28);
  color: rgba(255,255,255,0.96);
  font-size: 13px;
  font-weight: 700;
}

.rpl-muted { opacity: 0.78; font-size: 13px; }
.rpl-small { opacity: 0.82; font-size: 12px; line-height: 1.35; }

.kpi-number { color: #ffd54d; font-weight: 900; font-size: 34px; line-height: 1.05; }
.kpi-label { color: rgba(255,255,255,0.70); font-size: 13px; margin-top: 4px; }
hr { border: none; height: 1px; background: rgba(255,255,255,0.06); margin: 14px 0; }

@media (max-width: 768px) {
  .block-container { padding-left: 0.8rem !important; padding-right: 0.8rem !important; padding-top: 0.6rem !important; }
  .rpl-banner { padding: 22px 14px 16px 14px !important; border-radius: 16px !important; }
  .rpl-card { padding: 12px 12px !important; border-radius: 14px !important; }
  div[data-testid="stHorizontalBlock"] { flex-direction: column !important; }
  div[data-testid="column"] { width: 100% !important; flex: 1 1 100% !important; }
  div[data-testid="stTabs"] button { font-size: 12px !important; padding: 6px 10px !important; }
  div[data-testid="stTabs"] [role="tablist"] { overflow-x: auto !important; flex-wrap: nowrap !important; }
  div[data-testid="stTabs"] [role="tablist"]::-webkit-scrollbar { height: 6px; }
}
</style>
""",
    unsafe_allow_html=True,
)

# --- Constants ---
LAST_UPDATED_DROP = 25
TOTAL_VALID_DROPS = 24  # Drop 12 scrapped

ARCH_DEF = {
    "Strategist": "High participation + deliberate Power Play usage. Plays the long game and waits for leverage.",
    "Anchor": "Low volatility. Stays steady and doesn’t flinch when the room swings.",
    "Maverick": "High contrarian rate. Comfortable standing alone with minority reads; may use Power Play contrarianly.",
    "Wildcard": "High-variance profile. Volatility + higher-risk positioning, often combined with early Power Play usage.",
    "Calibrator": "Frequent recalibration across drops; changes stance based on signals.",
    "Wiseman": "High consensus alignment. Amplifies collective clarity when confidence is strong.",
    "Pragmatist": "Balanced and situational. Mixes approaches without chasing extremes.",
    "Ghost": "Low attendance so far. Not enough signal yet to infer a stable style."
}

BUCKET_EXPLAIN = {
    "H": "Most popular option (crowd favourite)",
    "M": "Middle options (neither most nor least popular)",
    "L": "Least popular option (contrarian lane)"
}

# --- Helpers ---
def clean_name_strong(s: str) -> str:
    if pd.isna(s):
        return ""
    t = str(s).replace("\u00A0", " ")
    t = re.sub(r"[^\w\s\-']", "", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t

def plotly_dark(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.92)"),
        margin=dict(l=8, r=8, t=35, b=10),
    )
    return fig

def compute_streaks(flags):
    longest = cur = 0
    for f in flags:
        if f:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    current = 0
    for f in reversed(flags):
        if f:
            current += 1
        else:
            break
    return longest, current

def parse_master_drop_groups(raw: pd.DataFrame):
    """
    master_data.xlsx / Summary format:
    Row 1 (index 0) contains Drop headers: "Drop 1", "Drop 2", ...
    Row 3 (index 2) contains question text for the response column of each drop.
    Each drop block width:
      - Drop 1..11: 2 columns (Response, Bucket)
      - Drop 12: scrapped (ignore)
      - Drop 13+: 3 columns (Response, PP, Bucket)
    """
    row0 = raw.iloc[0].astype(str)
    starts = [(i, row0[i]) for i in range(len(row0)) if isinstance(row0[i], str) and row0[i].startswith("Drop")]
    groups = []
    for idx, (start, label) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else raw.shape[1]
        width = end - start
        m = re.search(r"Drop\s+(\d+)", label)
        drop_num = int(m.group(1)) if m else None
        if drop_num is None or drop_num == 12:
            continue
        resp_col = start
        q_text = raw.iloc[2, resp_col] if raw.shape[0] > 2 else f"Q{drop_num}"
        q_text = str(q_text).strip() if pd.notna(q_text) else f"Q{drop_num}"
        if width == 2:
            pp_col = None
            bucket_col = start + 1
        else:
            pp_col = start + 1
            bucket_col = start + 2
        groups.append(dict(drop=drop_num, resp_col=resp_col, pp_col=pp_col, bucket_col=bucket_col, question=q_text))
    return sorted(groups, key=lambda x: x["drop"])

# --- Banner ---
st.markdown(
    f"""
<div class="rpl-banner">
  <div style="font-size: 32px; font-weight: 900; letter-spacing: 0.2px;">RPL 6 Experience Centre</div>
  <div style="opacity: 0.95; margin-top: 6px; font-size: 15px;">
    A behavioral analysis of each player's predictions and patterns
    <span class="rpl-pill" style="margin-left:12px;">Last updated: Drop {LAST_UPDATED_DROP}</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# --- Load baked files ---
DATA_DIR = Path("data")
metrics_path = DATA_DIR / "drop25_metrics.xlsx"
master_path = DATA_DIR / "master_data.xlsx"

if not metrics_path.exists() or not master_path.exists():
    st.error("Missing data files. Add to repo under `data/`: drop25_metrics.xlsx and master_data.xlsx")
    st.stop()

metrics = pd.read_excel(metrics_path, header=0)

# Flexible column mapping (case-insensitive)
cols_lower = {str(c).strip().lower(): c for c in metrics.columns}
required = ["player","unique one-liner (editable)","attended","h%","m%","l%","volatility%","risk score","pp taken","archetype"]
missing = [c for c in required if c not in cols_lower]
if missing:
    st.error(f"Metrics file missing columns: {missing}\nFound: {list(metrics.columns)}")
    st.stop()

metrics["Player"] = metrics[cols_lower["player"]].astype(str).apply(lambda x: re.sub(r"\s+", " ", x).strip())
metrics["player_key"] = metrics["Player"].apply(clean_name_strong)
metrics["Archetype"] = metrics[cols_lower["archetype"]].astype(str)

metrics["Attended_out_of_24"] = pd.to_numeric(metrics[cols_lower["attended"]], errors="coerce").fillna(0).astype(int)
metrics["AttendancePct_out_of_24"] = metrics["Attended_out_of_24"] / TOTAL_VALID_DROPS * 100.0

metrics["H%"] = pd.to_numeric(metrics[cols_lower["h%"]], errors="coerce").fillna(0.0)
metrics["M%"] = pd.to_numeric(metrics[cols_lower["m%"]], errors="coerce").fillna(0.0)
metrics["L%"] = pd.to_numeric(metrics[cols_lower["l%"]], errors="coerce").fillna(0.0)
metrics["Volatility%"] = pd.to_numeric(metrics[cols_lower["volatility%"]], errors="coerce").fillna(0.0)
metrics["Risk Score"] = pd.to_numeric(metrics[cols_lower["risk score"]], errors="coerce").fillna(0.0)
metrics["PP Taken"] = pd.to_numeric(metrics[cols_lower["pp taken"]], errors="coerce").fillna(0).astype(int)
metrics["Unique one-liner (editable)"] = metrics[cols_lower["unique one-liner (editable)"]].fillna("").astype(str)

# --- Load master data for Power Play moments + streaks ---
raw_master = pd.read_excel(master_path, sheet_name="Summary", header=None)
groups = parse_master_drop_groups(raw_master)
q_lookup = {g["drop"]: g["question"] for g in groups}

# Build row map for player name in master (col C -> index 2)
master_key_to_row = {}
for rr in range(3, raw_master.shape[0]):
    nm = raw_master.iloc[rr, 2] if raw_master.shape[1] > 2 else None
    if nm is None or (isinstance(nm, float) and np.isnan(nm)):
        continue
    master_key_to_row[clean_name_strong(nm)] = rr
master_keys = list(master_key_to_row.keys())

pp_details = {}
streaks = {}
valid_pp_second = {}

for _, row in metrics.iterrows():
    pk = row["player_key"]
    r = master_key_to_row.get(pk)
    if r is None:
        cand = difflib.get_close_matches(pk, master_keys, n=1, cutoff=0.78)
        if cand:
            r = master_key_to_row.get(cand[0])

    flags = []
    pp_yes = []
    pp_bucket = {}

    if r is not None:
        for g in groups:
            resp = raw_master.iloc[r, g["resp_col"]]
            answered = not (pd.isna(resp) or (isinstance(resp, str) and str(resp).strip() == ""))
            flags.append(answered)

            b = raw_master.iloc[r, g["bucket_col"]]
            b = b.strip() if isinstance(b, str) else ""

            if g["pp_col"] is not None:
                pp = raw_master.iloc[r, g["pp_col"]]
                # robust yes detection
                if isinstance(pp, str) and pp.strip().lower() == "yes":
                    pp_yes.append(g["drop"])
                    if answered and b in ["H","M","L"]:
                        pp_bucket[g["drop"]] = b
    else:
        flags = [False] * len(groups)

    longest, current = compute_streaks(flags)
    streaks[pk] = (longest, current)

    pp_total = len(pp_yes)
    pp_valid = pp_yes[:2]

    moments = []
    if len(pp_valid) >= 1:
        d1 = int(pp_valid[0])
        moments.append(dict(ordinal="First", drop=d1, question=q_lookup.get(d1, f"Q{d1}"), bucket=pp_bucket.get(d1, "")))
    if len(pp_valid) >= 2:
        d2 = int(pp_valid[1])
        moments.append(dict(ordinal="Second", drop=d2, question=q_lookup.get(d2, f"Q{d2}"), bucket=pp_bucket.get(d2, "")))
        valid_pp_second[pk] = d2

    pp_details[pk] = dict(pp_total_marked=pp_total, moments=moments)

metrics["LongestStreak"] = metrics["player_key"].map(lambda k: streaks.get(k, (np.nan, np.nan))[0])
metrics["CurrentStreak"] = metrics["player_key"].map(lambda k: streaks.get(k, (np.nan, np.nan))[1])
metrics["Second_PP_by_Q"] = metrics["player_key"].map(lambda k: valid_pp_second.get(k, np.nan))

# =========================
# The Room
# =========================
st.markdown("## The Room")
st.markdown('<div class="rpl-muted">Behavioral shape of the room based on H/M/L buckets, volatility, and Power Play usage (Drop 12 excluded).</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c5, c6, c7, c8 = st.columns(4)

total_responses = int(metrics["Attended_out_of_24"].sum())
active_25 = int((metrics["AttendancePct_out_of_24"] >= 25).sum())
perfect = int((metrics["Attended_out_of_24"] == TOTAL_VALID_DROPS).sum())
pp_exhausted = int((metrics["PP Taken"] >= 2).sum())

total_bucketed = total_responses if total_responses else 1
room_H = float((metrics["H%"] / 100.0 * metrics["Attended_out_of_24"]).sum() / total_bucketed * 100.0)

with c1: st.markdown(f"<div class='kpi-number'>{metrics['Player'].nunique()}</div><div class='kpi-label'>Players</div>", unsafe_allow_html=True)
with c2: st.markdown(f"<div class='kpi-number'>{active_25}</div><div class='kpi-label'>Active (≥25% attendance)</div>", unsafe_allow_html=True)
with c3: st.markdown(f"<div class='kpi-number'>{TOTAL_VALID_DROPS}</div><div class='kpi-label'>Total valid drops</div>", unsafe_allow_html=True)
with c4: st.markdown(f"<div class='kpi-number'>{total_responses}</div><div class='kpi-label'>Total responses</div>", unsafe_allow_html=True)

with c5: st.markdown(f"<div class='kpi-number'>{perfect}</div><div class='kpi-label'>Perfect Attendance Club</div>", unsafe_allow_html=True)
with c6:
    streak10 = int((metrics["CurrentStreak"] >= 10).sum())
    st.markdown(f"<div class='kpi-number'>{streak10}</div><div class='kpi-label'>10+ Current Streak Club</div>", unsafe_allow_html=True)
    st.markdown("<div class='rpl-small rpl-muted'># players who have not missed the previous 10+ drops</div>", unsafe_allow_html=True)
with c7: st.markdown(f"<div class='kpi-number'>{pp_exhausted}</div><div class='kpi-label'>PP exhausted (≥2 marked)</div>", unsafe_allow_html=True)
with c8: st.markdown(f"<div class='kpi-number'>{room_H:.1f}%</div><div class='kpi-label'>Room consensus tilt (H share)</div>", unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# Metric explainer ABOVE charts (as you asked)
with st.expander("Metric explainer (tap to open)", expanded=False):
    st.markdown(
        """
- **H% / M% / L%**: Share of answered picks in the Most popular / Middle / Least popular bucket.
- **Risk Score**: Average distance from consensus (**H=0, M=0.5, L=1**).
- **Volatility%**: % of answered drops where your bucket changed vs previous answered drop.
- **Attendance**: Answered valid drops out of **24**.
- **Power Play**: Only first **two** count.
"""
    )
    st.markdown("### Archetypes (simple definitions)")
    for k in ["Strategist","Anchor","Wiseman","Maverick","Calibrator","Wildcard","Pragmatist","Ghost"]:
        st.markdown(f"**{k}** — {ARCH_DEF[k]}")

# Charts
cc1, cc2 = st.columns([1.2, 1.0])

with cc1:
    H_overall = float((metrics["H%"]/100.0 * metrics["Attended_out_of_24"]).sum())
    M_overall = float((metrics["M%"]/100.0 * metrics["Attended_out_of_24"]).sum())
    L_overall = float((metrics["L%"]/100.0 * metrics["Attended_out_of_24"]).sum())
    mix_df = pd.DataFrame({
        "Bucket": ["H", "M", "L"],
        "Count": [H_overall, M_overall, L_overall],
        "Meaning": [BUCKET_EXPLAIN["H"], BUCKET_EXPLAIN["M"], BUCKET_EXPLAIN["L"]],
    })
    fig = px.pie(mix_df, values="Count", names="Bucket", hole=0.55, hover_data=["Meaning"])
    fig.update_traces(textposition="inside", textinfo="percent+label")
    # fix legend overlap by pushing it below chart
    fig.update_layout(legend=dict(orientation="h", y=-0.18, x=0.2))
    fig = plotly_dark(fig)
    fig.update_layout(title="Overall H / M / L mix (weighted by participation)", height=420)
    st.plotly_chart(fig, use_container_width=True)

with cc2:
    arch_counts = metrics["Archetype"].value_counts().reset_index()
    arch_counts.columns = ["Archetype", "Players"]
    mapping = metrics.groupby("Archetype")["Player"].apply(lambda x: ", ".join(x.tolist())).to_dict()
    arch_counts["PlayersList"] = arch_counts["Archetype"].map(mapping)
    fig2 = px.bar(arch_counts, x="Archetype", y="Players", text="Players")
    fig2.update_traces(
        hovertemplate="%{x}<br>%{y} players<br><br>%{customdata[0]}",
        customdata=arch_counts[["PlayersList"]].values
    )
    fig2 = plotly_dark(fig2)
    fig2.update_layout(title="Archetype distribution (hover to see names)", height=420)
    st.plotly_chart(fig2, use_container_width=True)

# =========================
# Player Experience Centre
# =========================
st.markdown("## Player Experience Centre")
st.markdown('<div class="rpl-muted">Pick a player to see: style, streaks, H/M/L mix, and Power Play moments.</div>', unsafe_allow_html=True)

sel = st.selectbox("Select player", metrics["Player"].tolist(), index=0)
p = metrics.loc[metrics["Player"] == sel].iloc[0]
pk = p["player_key"]

l, m, r = st.columns([1.15, 1.0, 1.15])

with l:
    st.markdown('<div class="rpl-card">', unsafe_allow_html=True)
    st.markdown(f"### {p['Player']}")
    st.markdown(f"<span class='rpl-pill'>{p['Archetype']}</span>", unsafe_allow_html=True)
    st.markdown(f"<div class='rpl-small' style='margin-top:8px;'><b>Archetype logic:</b> {ARCH_DEF.get(p['Archetype'], '')}</div>", unsafe_allow_html=True)
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("**What's unique about you**")
    st.write(p["Unique one-liner (editable)"])
    st.markdown("</div>", unsafe_allow_html=True)

with m:
    st.markdown('<div class="rpl-card">', unsafe_allow_html=True)
    st.markdown("### Your scoreboard")
    st.metric("Attendance (out of 24)", int(p["Attended_out_of_24"]))
    st.progress(min(max(float(p["AttendancePct_out_of_24"]) / 100.0, 0.0), 1.0))
    st.markdown(f"<div class='rpl-muted'>Power Plays marked: <b>{int(p['PP Taken'])}</b> / 2 count</div>", unsafe_allow_html=True)
    st.progress(min(int(p["PP Taken"]) / 2.0, 1.0))
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.metric("Current streak", int(p["CurrentStreak"]))
    st.metric("Longest streak", int(p["LongestStreak"]))
    st.markdown("</div>", unsafe_allow_html=True)

with r:
    st.markdown('<div class="rpl-card">', unsafe_allow_html=True)
    st.markdown("### Style signals")
    pmix = pd.DataFrame({
        "Bucket": ["H", "M", "L"],
        "Share": [float(p["H%"]), float(p["M%"]), float(p["L%"])],
        "Meaning": [BUCKET_EXPLAIN["H"], BUCKET_EXPLAIN["M"], BUCKET_EXPLAIN["L"]],
    })
    figp = px.pie(pmix, values="Share", names="Bucket", hole=0.55, hover_data=["Meaning"])
    figp.update_traces(textposition="inside", textinfo="percent+label")
    figp = plotly_dark(figp)
    figp.update_layout(title="Your H / M / L mix", height=360)
    st.plotly_chart(figp, use_container_width=True)
    st.markdown(f"<div class='rpl-small'><b>Risk Score:</b> {float(p['Risk Score']):.2f} &nbsp; • &nbsp; <b>Volatility:</b> {float(p['Volatility%']):.1f}%</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Power Play moments
# =========================
st.markdown("### Power Play moments")
st.markdown('<div class="rpl-muted">Only the first two Power Plays count. If you marked more, your early Power Play moments are the ones that matter.</div>', unsafe_allow_html=True)

info = pp_details.get(pk, {"pp_total_marked": 0, "moments": []})
pp_total = info.get("pp_total_marked", 0)
moments = info.get("moments", [])

if pp_total == 0:
    st.markdown('<div class="rpl-card">No Power Play moments recorded yet.</div>', unsafe_allow_html=True)
else:
    if not moments:
        st.markdown('<div class="rpl-card">Power Plays detected, but could not map them to master questions.</div>', unsafe_allow_html=True)
    else:
        box = "<div class='rpl-card'>"
        for mo in moments:
            lbl = {"H": "crowd pick", "M": "balanced pick", "L": "contrarian pick"}.get(mo.get("bucket", ""), "")
            extra = f"— you took a <b>{lbl}</b> on this one." if lbl else ""
            box += (
                f"<div style='margin-bottom:12px;'>"
                f"<b>{mo['ordinal']} Power Play</b> on <b>Q{mo['drop']}</b>:<br/>"
                f"{mo['question']}<br/>"
                f"<span class='rpl-muted'>{extra}</span>"
                f"</div>"
            )
        if pp_total > 2:
            box += f"<div class='rpl-small rpl-muted'>You marked Power Play <b>{pp_total}</b> times — only the first two count.</div>"
        box += "</div>"
        st.markdown(box, unsafe_allow_html=True)

# =========================
# Risk + Story Mode
# =========================
st.markdown("## Risk + Power Plays + Story Mode")

st.markdown("### Risk vs Attendance Map")
st.markdown('<div class="rpl-muted">Y-axis: Risk Score (H→L). X-axis: Attendance%. Hover: name + archetype only.</div>', unsafe_allow_html=True)

map_df = metrics.copy()
map_df["HoverName"] = map_df.apply(lambda rr: f"{rr['Player']}; {rr['Archetype']}", axis=1)

figm = px.scatter(
    map_df,
    x="AttendancePct_out_of_24",
    y="Risk Score",
    color="Archetype",
    hover_name="HoverName",
)
figm.update_traces(marker=dict(size=10, line=dict(width=0.5, color="rgba(255,255,255,0.14)")))
figm = plotly_dark(figm)
figm.update_layout(height=520)
st.plotly_chart(figm, use_container_width=True)

st.markdown("### Power Play Status")
pp0 = metrics[metrics["PP Taken"] == 0]["Player"].tolist()
pp1 = metrics[metrics["PP Taken"] == 1]["Player"].tolist()
pp2p = metrics[metrics["PP Taken"] >= 2]["Player"].tolist()

b1, b2, b3 = st.columns(3)
with b1:
    st.markdown('<div class="rpl-card">', unsafe_allow_html=True)
    st.markdown(f"**Conviction still loaded ({len(pp0)})**")
    st.write(", ".join(pp0) if pp0 else "—")
    st.markdown("</div>", unsafe_allow_html=True)
with b2:
    st.markdown('<div class="rpl-card">', unsafe_allow_html=True)
    st.markdown(f"**One-shot conviction ({len(pp1)})**")
    st.write(", ".join(pp1) if pp1 else "—")
    st.markdown("</div>", unsafe_allow_html=True)
with b3:
    st.markdown('<div class="rpl-card">', unsafe_allow_html=True)
    st.markdown(f"**Two chips spent ({len(pp2p)})**")
    st.write(", ".join(pp2p) if pp2p else "—")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("### Story Mode")
tabs = st.tabs(["Contrarian League", "Most Volatile", "Wildcards", "Conviction Still Loaded", "All-In Early"])
story = metrics[(metrics["AttendancePct_out_of_24"] >= 30) & (metrics["Attended_out_of_24"] > 0)].copy()

with tabs[0]:
    st.markdown('<div class="rpl-card">', unsafe_allow_html=True)
    st.markdown("**What this list means:** Players who most often choose the least popular option (high L%).")
    top = story.sort_values("L%", ascending=False).head(10)
    st.write(", ".join(top["Player"].tolist()) if len(top) else "—")
    st.markdown("</div>", unsafe_allow_html=True)

with tabs[1]:
    st.markdown('<div class="rpl-card">', unsafe_allow_html=True)
    st.markdown("**What this list means:** Players who change buckets frequently (high volatility).")
    top = story.sort_values("Volatility%", ascending=False).head(10)
    st.write(", ".join(top["Player"].tolist()) if len(top) else "—")
    st.markdown("</div>", unsafe_allow_html=True)

with tabs[2]:
    st.markdown('<div class="rpl-card">', unsafe_allow_html=True)
    st.markdown("**What this list means:** Wildcards are high-variance profiles (risk + volatility). Different from Contrarians: contrarians stay L-leaning; wildcards can swing anywhere.")
    top = story[story["Archetype"] == "Wildcard"].sort_values(["AttendancePct_out_of_24", "Risk Score"], ascending=False).head(12)
    st.write(", ".join(top["Player"].tolist()) if len(top) else "—")
    st.markdown("</div>", unsafe_allow_html=True)

with tabs[3]:
    st.markdown('<div class="rpl-card">', unsafe_allow_html=True)
    st.markdown("**What this list means:** Players who haven’t used a Power Play yet — both chips still available.")
    top = story[story["PP Taken"] == 0].sort_values("AttendancePct_out_of_24", ascending=False).head(20)
    st.write(", ".join(top["Player"].tolist()) if len(top) else "—")
    st.markdown("</div>", unsafe_allow_html=True)

with tabs[4]:
    st.markdown('<div class="rpl-card">', unsafe_allow_html=True)
    st.markdown("**What this list means:** Players who exhausted both valid Power Plays early — we show the question by which their 2nd PP was used.")
    tmp = metrics.dropna(subset=["Second_PP_by_Q"]).sort_values("Second_PP_by_Q", ascending=True).head(12)
    items = [f"{rr.Player} (by Q{int(rr.Second_PP_by_Q)})" for rr in tmp.itertuples()]
    st.write(", ".join(items) if items else "—")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("### Perfect Attendance Club (24/24)")
club = metrics[metrics["Attended_out_of_24"] == TOTAL_VALID_DROPS]["Player"].tolist()
st.markdown('<div class="rpl-card">', unsafe_allow_html=True)
st.write(", ".join(club) if club else "—")
st.markdown("</div>", unsafe_allow_html=True)
