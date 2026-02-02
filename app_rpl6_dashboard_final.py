import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re, difflib
from pathlib import Path

# ===========================================
# RPL 6 Experience Centre — Drop 33 build
# - Metrics computed from master_data.xlsx
# - One-liners re-used from prior metrics file (drop25_metrics.xlsx) if present
#
# Required repo file:
#   data/master_data.xlsx
# Optional repo file (for narratives only):
#   data/drop25_metrics.xlsx
# ===========================================

st.set_page_config(page_title="RPL 6 Experience Centre", layout="wide")

# ---------- Styling ----------
st.markdown(
    """
<style>
html, body, [data-testid="stAppViewContainer"] { background: #0b0f14 !important; color: rgba(255,255,255,0.92) !important; }
section[data-testid="stSidebar"] > div { background: #0b0f14 !important; border-right: 1px solid rgba(255,255,255,0.06); }
.block-container { padding-top: 1.0rem; padding-bottom: 2.0rem; max-width: 1200px; }
h1,h2,h3,h4 { color: rgba(255,255,255,0.96) !important; }

.rpl-banner {
  border-radius: 22px;
  padding: 32px 22px 26px 22px;
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

.yellow-label { color:#ffd54d !important; font-weight:800; }
.yellow-metric { color:#ffd54d !important; font-weight:900; font-size:34px; }
.yellow-submetric { color:#ffd54d !important; font-weight:900; font-size:22px; }

.ribbon {
  border-radius: 14px;
  padding: 10px 12px;
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.06);
  margin-bottom: 10px;
  color: rgba(255,255,255,0.90);
  font-weight: 800;
}

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
BUCKET_SCORE = {"H": 0.0, "M": 0.5, "L": 1.0}

def clean_name(s: str) -> str:
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
    FIXED:
    - Question text comes from row index 1 (your file’s question row)
    - Drop headers are in row index 0
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
        # question row is index 1
        q_text = raw.iloc[1, resp_col] if raw.shape[0] > 1 else f"Q{drop_num}"
        q_text = str(q_text).strip() if pd.notna(q_text) else f"Q{drop_num}"
        if width == 2:
            pp_col = None
            bucket_col = start + 1
        else:
            pp_col = start + 1
            bucket_col = start + 2
        groups.append(dict(drop=drop_num, resp_col=resp_col, pp_col=pp_col, bucket_col=bucket_col, question=q_text))
    groups = sorted(groups, key=lambda x: x["drop"])
    max_drop = max([g["drop"] for g in groups]) if groups else 0
    return groups, max_drop

def classify_archetype(att_pct, H_pct, M_pct, L_pct, vol_pct, risk, pp_total):
    if att_pct < 25:
        return "Ghost"
    if (L_pct >= 30) and (risk >= 0.65):
        return "Maverick"
    if (vol_pct >= 60) and (risk >= 0.60):
        return "Wildcard"
    if (H_pct >= 55) and (risk <= 0.25):
        return "Wiseman"
    if (vol_pct >= 55) and (0.30 <= risk <= 0.60):
        return "Calibrator"
    if (vol_pct <= 25) and (risk <= 0.35):
        return "Anchor"
    if (att_pct >= 80) and (pp_total >= 1):
        return "Strategist"
    return "Pragmatist"

# ------------------- Load data -------------------
DATA_DIR = Path("data")
master_path = DATA_DIR / "master_data.xlsx"
narr_path = DATA_DIR / "drop25_metrics.xlsx"  # optional, for one-liners only

if not master_path.exists():
    st.error("Missing master file: data/master_data.xlsx (rename your uploaded master to this exact name)")
    st.stop()

raw_master = pd.read_excel(master_path, sheet_name="Summary", header=None)
groups, last_drop = parse_master_drop_groups(raw_master)
TOTAL_VALID_DROPS = len(groups)
q_text_by_drop = {g["drop"]: g["question"] for g in groups}

# Optional narratives
one_liners = {}
if narr_path.exists():
    try:
        narr = pd.read_excel(narr_path)
        cols = {str(c).strip().lower(): c for c in narr.columns}
        pcol = cols.get("player")
        lcol = cols.get("unique one-liner (editable)") or cols.get("unique one-liner") or cols.get("one liner") or cols.get("one-liner")
        if pcol and lcol:
            tmp = narr[[pcol, lcol]].copy()
            tmp[pcol] = tmp[pcol].astype(str)
            tmp["key"] = tmp[pcol].apply(clean_name)
            for _, rr in tmp.iterrows():
                one_liners[rr["key"]] = "" if pd.isna(rr[lcol]) else str(rr[lcol]).strip()
    except Exception:
        one_liners = {}

# Build player list from master col C (index 2)
# FIXED: start at row index 2 (so Ananth doesn't get skipped)
players = []
row_map = {}
for rr in range(2, raw_master.shape[0]):
    nm = raw_master.iloc[rr, 2] if raw_master.shape[1] > 2 else None
    if nm is None or (isinstance(nm, float) and np.isnan(nm)):
        continue
    name = re.sub(r"\s+", " ", str(nm)).strip()
    k = clean_name(name)
    if k:
        players.append(name)
        row_map[k] = rr

players = sorted(list(dict.fromkeys(players)), key=lambda x: x.lower())

# ------------------- Compute metrics from master -------------------
rows = []
pp_details = {}
valid_pp_second = {}

for name in players:
    pk = clean_name(name)
    rr = row_map.get(pk)
    if rr is None:
        continue

    answered_flags = []
    bucket_seq = []
    pp_yes_drops = []
    pp_bucket = {}

    for g in groups:
        resp = raw_master.iloc[rr, g["resp_col"]]
        answered = not (pd.isna(resp) or (isinstance(resp, str) and str(resp).strip() == ""))
        answered_flags.append(answered)

        b = raw_master.iloc[rr, g["bucket_col"]]
        b = b.strip() if isinstance(b, str) else ""
        if answered and b in BUCKET_SCORE:
            bucket_seq.append(b)

        if g["pp_col"] is not None:
            pp = raw_master.iloc[rr, g["pp_col"]]
            if isinstance(pp, str) and pp.strip().lower() == "yes":
                pp_yes_drops.append(g["drop"])
                if answered and b in ["H", "M", "L"]:
                    pp_bucket[g["drop"]] = b

    attended = int(sum(answered_flags))
    att_pct = (attended / TOTAL_VALID_DROPS * 100.0) if TOTAL_VALID_DROPS else 0.0

    if attended > 0 and bucket_seq:
        H_ct = sum(1 for x in bucket_seq if x == "H")
        M_ct = sum(1 for x in bucket_seq if x == "M")
        L_ct = sum(1 for x in bucket_seq if x == "L")
        denom = max(1, (H_ct + M_ct + L_ct))
        H_pct = H_ct / denom * 100.0
        M_pct = M_ct / denom * 100.0
        L_pct = L_ct / denom * 100.0
        risk = float(np.mean([BUCKET_SCORE[x] for x in bucket_seq]))
    else:
        H_pct = M_pct = L_pct = 0.0
        risk = 0.0

    if len(bucket_seq) <= 1:
        vol = 0.0
    else:
        changes = sum(1 for i in range(1, len(bucket_seq)) if bucket_seq[i] != bucket_seq[i-1])
        vol = changes / (len(bucket_seq) - 1) * 100.0

    longest, current = compute_streaks(answered_flags)

    pp_total = len(pp_yes_drops)
    pp_valid = pp_yes_drops[:2]
    moments = []
    if len(pp_valid) >= 1:
        d1 = int(pp_valid[0])
        moments.append(dict(ordinal="First", drop=d1, question=q_text_by_drop.get(d1, f"Q{d1}"), bucket=pp_bucket.get(d1, "")))
    if len(pp_valid) >= 2:
        d2 = int(pp_valid[1])
        moments.append(dict(ordinal="Second", drop=d2, question=q_text_by_drop.get(d2, f"Q{d2}"), bucket=pp_bucket.get(d2, "")))
        valid_pp_second[pk] = d2

    pp_details[pk] = dict(pp_total_marked=pp_total, moments=moments)

    archetype = classify_archetype(att_pct, H_pct, M_pct, L_pct, vol, risk, pp_total)

    one_liner = one_liners.get(pk, "")
    if not one_liner:
        one_liner = "Narrative not loaded yet (will be updated next)."

    rows.append({
        "Player": name,
        "player_key": pk,
        "Archetype": archetype,
        "Attended": attended,
        "AttendancePct": att_pct,
        "H%": H_pct,
        "M%": M_pct,
        "L%": L_pct,
        "Volatility%": vol,
        "Risk Score": risk,
        "PP Taken": pp_total,
        "LongestStreak": longest,
        "CurrentStreak": current,
        "Second_PP_by_Q": valid_pp_second.get(pk, np.nan),
        "OneLiner": one_liner
    })

metrics = pd.DataFrame(rows)
if metrics.empty:
    st.error("Could not compute metrics from master_data.xlsx. Check the sheet name is 'Summary' and player names exist in column C.")
    st.stop()

# ------------------- Banner -------------------
st.markdown(
    f"""
<div class="rpl-banner">
  <div style="font-size: 32px; font-weight: 900; letter-spacing: 0.2px;">RPL 6 Experience Centre</div>
  <div style="opacity: 0.95; margin-top: 6px; font-size: 15px;">
    A behavioral analysis of each player's predictions and patterns
    <span class="rpl-pill" style="margin-left:12px;">Last updated: Drop {last_drop}</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ------------------- The Room -------------------
st.markdown("## The Room")
st.markdown('<div class="rpl-muted">Behavioral shape of the room based on H/M/L buckets, volatility, and Power Play usage (Drop 12 excluded).</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c5, c6, c7, c8 = st.columns(4)

total_responses = int(metrics["Attended"].sum())
active_25 = int((metrics["AttendancePct"] >= 25).sum())
perfect = int((metrics["Attended"] == TOTAL_VALID_DROPS).sum())
pp_exhausted = int((metrics["PP Taken"] >= 2).sum())

total_bucketed = total_responses if total_responses else 1
room_H = float(((metrics["H%"] / 100.0) * metrics["Attended"]).sum() / total_bucketed * 100.0)

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

# ------------------- Player Experience Centre -------------------
st.markdown("## Player Experience Centre")
st.markdown('<div class="rpl-muted">Pick a player to see: style, streaks, H/M/L mix, and Power Play moments.</div>', unsafe_allow_html=True)

st.markdown("<div class='yellow-label'>Select player</div>", unsafe_allow_html=True)
sel = st.selectbox("", metrics["Player"].tolist(), index=0, label_visibility="collapsed")
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
    st.write(p["OneLiner"])
    st.markdown("</div>", unsafe_allow_html=True)

with m:
    st.markdown('<div class="rpl-card">', unsafe_allow_html=True)
    st.markdown("### Your scoreboard")
    st.markdown(f"<div class='yellow-metric'>{int(p['Attended'])}</div><div class='yellow-label'>Attendance (out of {TOTAL_VALID_DROPS})</div>", unsafe_allow_html=True)
    st.progress(min(max(float(p["AttendancePct"]) / 100.0, 0.0), 1.0))

    st.markdown(f"<div class='yellow-label' style='margin-top:10px;'>Power Plays marked: {int(p['PP Taken'])} / 2 count</div>", unsafe_allow_html=True)
    st.progress(min(int(p["PP Taken"]) / 2.0, 1.0))

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(f"<div class='yellow-submetric'>{int(p['CurrentStreak'])}</div><div class='yellow-label'>Current streak</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='yellow-submetric' style='margin-top:10px;'>{int(p['LongestStreak'])}</div><div class='yellow-label'>Longest streak</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with r:
    st.markdown('<div class="rpl-card">', unsafe_allow_html=True)
    st.markdown("### Style signals")

    H = float(p["H%"]); Mv = float(p["M%"]); L = float(p["L%"])

    # stacked vertical bar
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[""], y=[L], name="L", marker=dict(color="#34d399")))
    fig.add_trace(go.Bar(x=[""], y=[Mv], name="M", marker=dict(color="#ef4444")))
    fig.add_trace(go.Bar(x=[""], y=[H], name="H", marker=dict(color="#6366f1")))
    fig.update_layout(barmode="stack", showlegend=False, height=380)
    fig = plotly_dark(fig)
    fig.update_yaxes(range=[0, 100], title="")
    fig.update_xaxes(showticklabels=False, title="")

    # arrows + labels on RHS using annotations (paper coords)
    # We'll place them at approx segment midpoints
    yL = L/200
    yM = (L + Mv/2)/100
    yH = (L + Mv + H/2)/100

    def anno(y, text):
        fig.add_annotation(
            x=1.04, y=y, xref="paper", yref="paper",
            text=text,
            showarrow=True, arrowhead=2, ax=30, ay=0,
            font=dict(color="#ffd54d", size=12),
            arrowcolor="rgba(255,255,255,0.40)"
        )

    anno(yH, f"H: crowd picks — {H:.0f}%")
    anno(yM, f"M: middle picks — {Mv:.0f}%")
    anno(yL, f"L: contrarian picks — {L:.0f}%")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"""
<div style="display:flex; gap:18px; margin-top:6px;">
  <div>
    <div class="yellow-label">🎯 Risk Score</div>
    <div class="yellow-submetric">{float(p['Risk Score']):.2f}</div>
  </div>
  <div>
    <div class="yellow-label">🌊 Volatility</div>
    <div class="yellow-submetric">{float(p['Volatility%']):.1f}%</div>
  </div>
</div>
""",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------- Power Play moments -------------------
st.markdown("### Power Play moments")
st.markdown('<div class="rpl-muted">Only the first two Power Plays count. If you marked more, your early Power Play moments are the ones that matter.</div>', unsafe_allow_html=True)

info = pp_details.get(pk, {"pp_total_marked": 0, "moments": []})
pp_total = info.get("pp_total_marked", 0)
moments = info.get("moments", [])

if pp_total == 0:
    st.markdown('<div class="rpl-card">No Power Play moments recorded yet.</div>', unsafe_allow_html=True)
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

# ------------------- Risk + Story Mode -------------------
st.markdown("## Risk + Power Plays + Story Mode")
st.markdown("### Risk vs Attendance Map")
st.markdown('<div class="rpl-muted">Y-axis: Risk Score (H→L). X-axis: Attendance%. Hover: name + archetype only.</div>', unsafe_allow_html=True)

map_df = metrics.copy()
map_df["HoverName"] = map_df.apply(lambda rr: f"{rr['Player']}; {rr['Archetype']}", axis=1)

figm = px.scatter(
    map_df,
    x="AttendancePct",
    y="Risk Score",
    color="Archetype",
    hover_name="HoverName",
)
figm.update_traces(marker=dict(size=10, line=dict(width=0.5, color="rgba(255,255,255,0.14)")))
figm = plotly_dark(figm)
figm.update_layout(height=520)
st.plotly_chart(figm, use_container_width=True)

# ------------------- Power Play Status (with ribbon text) -------------------
st.markdown("### Power Play Status")

pp0 = metrics[metrics["PP Taken"] == 0]["Player"].tolist()
pp1 = metrics[metrics["PP Taken"] == 1]["Player"].tolist()
pp2p = metrics[metrics["PP Taken"] >= 2]["Player"].tolist()

b1, b2, b3 = st.columns(3)

with b1:
    st.markdown('<div class="rpl-card">', unsafe_allow_html=True)
    st.markdown('<div class="ribbon">Both Power Plays not used yet</div>', unsafe_allow_html=True)
    st.markdown(f"**Conviction still loaded ({len(pp0)})**")
    st.write(", ".join(pp0) if pp0 else "—")
    st.markdown("</div>", unsafe_allow_html=True)

with b2:
    st.markdown('<div class="rpl-card">', unsafe_allow_html=True)
    st.markdown('<div class="ribbon">Only 1 Power Play used so far</div>', unsafe_allow_html=True)
    st.markdown(f"**One-shot conviction ({len(pp1)})**")
    st.write(", ".join(pp1) if pp1 else "—")
    st.markdown("</div>", unsafe_allow_html=True)

with b3:
    st.markdown('<div class="rpl-card">', unsafe_allow_html=True)
    st.markdown('<div class="ribbon">Both Power Plays exhausted</div>', unsafe_allow_html=True)
    st.markdown(f"**Two chips spent ({len(pp2p)})**")
    st.write(", ".join(pp2p) if pp2p else "—")
    st.markdown("</div>", unsafe_allow_html=True)
