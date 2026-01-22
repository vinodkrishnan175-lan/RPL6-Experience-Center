import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

# ==============================
# Page + Theme (Lovable-dark, Option C)
# ==============================
st.set_page_config(page_title="RPL 6 Experience Centre", layout="wide")

st.markdown(
    """
<style>
html, body, [data-testid="stAppViewContainer"] { background: #0b0f14 !important; color: rgba(255,255,255,0.92) !important; }
section[data-testid="stSidebar"] > div { background: #0b0f14 !important; border-right: 1px solid rgba(255,255,255,0.06); }
h1,h2,h3,h4 { color: rgba(255,255,255,0.96) !important; }
a, a:visited { color: #2d7ff9 !important; }
.block-container { padding-top: 1.0rem; padding-bottom: 2.0rem; max-width: 1200px; }
.rpl-banner { border-radius: 22px; padding: 18px 18px 14px 18px;
  background: linear-gradient(90deg, rgba(45,127,249,0.20), rgba(255,255,255,0.05));
  border: 1px solid rgba(255,255,255,0.10);
  box-shadow: 0 12px 28px rgba(0,0,0,0.30);
  margin-bottom: 16px; }
.rpl-card { border-radius: 18px; padding: 14px 16px;
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.10);
  box-shadow: 0 10px 24px rgba(0,0,0,0.24); }
.rpl-pill { display: inline-block; padding: 4px 10px; border-radius: 999px;
  background: rgba(45,127,249,0.20); border: 1px solid rgba(45,127,249,0.28);
  color: rgba(255,255,255,0.92); font-size: 12px; }
.rpl-muted { opacity: 0.78; font-size: 13px; }
.rpl-small { opacity: 0.82; font-size: 12px; line-height: 1.35; }
hr { border: none; height: 1px; background: rgba(255,255,255,0.08); margin: 14px 0; }
</style>
""",
    unsafe_allow_html=True,
)

# ==============================
# Constants / Labels
# ==============================
LAST_UPDATED_DROP = 25
TOTAL_VALID_DROPS = 24  # Drop 12 scrapped

ARCH_DEF = {
    "Strategist": "High participation + selective conviction. Plays the long game and waits for leverage.",
    "Anchor": "Low volatility. Stays steady and doesn’t flinch when the room swings.",
    "Maverick": "High contrarian rate. Comfortable standing alone with minority reads.",
    "Wildcard": "High variance profile. Volatile + higher-risk positioning creates big upside.",
    "Calibrator": "High volatility. Recalibrates stance frequently based on context.",
    "Wiseman": "High consensus alignment. Amplifies collective clarity when confidence is strong.",
    "Pragmatist": "Balanced and situational. Mixes approaches without chasing extremes.",
    "Ghost": "Low attendance so far. Not enough signal yet to infer a stable style."
}

BUCKET_EXPLAIN = {
    "H": "Most popular option (crowd favourite)",
    "M": "Middle options (neither most nor least popular)",
    "L": "Least popular option (contrarian lane)"
}

# ==============================
# Helpers
# ==============================
def clean_name(s: str) -> str:
    return re.sub(r"\\s+", " ", str(s).strip())

def key_name(s: str) -> str:
    return clean_name(s).lower()

def is_blank(x) -> bool:
    return pd.isna(x) or (isinstance(x, str) and str(x).strip() == "")

def safe_read_excel(uploaded_file, sheet=None, header=0):
    try:
        return pd.read_excel(uploaded_file, sheet_name=sheet, header=header)
    except Exception as e:
        st.error(f"Could not read the Excel file. Error: {e}")
        st.stop()

def parse_master_drop_groups(raw: pd.DataFrame):
    """
    raw is headerless df from master, where row 0 contains 'Drop X' blocks.
    Each drop block is 2 columns (Drop 1-11) or 3 columns (Drop 13+ with PP).
    Drop 12 exists but is scrapped.
    """
    row0 = raw.iloc[0].astype(str)
    starts = [(i, row0[i]) for i in range(len(row0)) if isinstance(row0[i], str) and row0[i].startswith("Drop")]
    if not starts:
        st.error("No 'Drop X' headers found in row 1 of the master sheet. Please re-upload the correct template.")
        st.stop()

    groups = []
    for idx, (start, label) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else raw.shape[1]
        width = end - start
        m = re.search(r"Drop\\s+(\\d+)", label)
        drop_num = int(m.group(1)) if m else None
        if drop_num is None or drop_num == 12:
            continue

        response_col = start
        # Question text expected on row 3 (0-indexed 2) in response column
        q_text = raw.iloc[2, response_col] if raw.shape[0] > 2 else f"Q{drop_num}"
        q_text = str(q_text).strip() if pd.notna(q_text) else f"Q{drop_num}"

        if width == 2:
            pp_col = None
            bucket_col = start + 1
        else:
            pp_col = start + 1
            bucket_col = start + 2

        groups.append({
            "drop": drop_num,
            "resp_col": response_col,
            "pp_col": pp_col,
            "bucket_col": bucket_col,
            "question": q_text
        })

    return sorted(groups, key=lambda x: x["drop"])

def compute_streaks(answered_flags):
    longest = 0
    cur = 0
    for a in answered_flags:
        if a:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0

    current = 0
    for a in reversed(answered_flags):
        if a:
            current += 1
        else:
            break
    return longest, current

def bucket_label(b):
    return {"H": "crowd pick", "M": "balanced pick", "L": "contrarian pick"}.get(b, "")

def plotly_dark_defaults(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.92)"),
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig

# ==============================
# Header
# ==============================
st.markdown(
    f"""
<div class="rpl-banner">
  <div style="font-size: 26px; font-weight: 850; letter-spacing: 0.2px;">
    RPL 6 Experience Centre
  </div>
  <div style="opacity: 0.92; margin-top: 4px; font-size: 14px;">
    A behavioral analysis of each player's predictions and patterns
    <span class="rpl-pill" style="margin-left:10px;">Last updated: Drop {LAST_UPDATED_DROP}</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ==============================
# Inputs
# ==============================
with st.sidebar:
    st.header("Inputs")
    metrics_file = st.file_uploader("Upload final narrative + metrics (Drop 25 revised input)", type=["xlsx"])
    master_file = st.file_uploader("Upload master (for PP moments + streaks)", type=["xlsx"])
    show_data = st.toggle("Show raw tables (debug)", value=False)

if metrics_file is None:
    st.info("Upload the **Drop 25 revised input** file (final narrative + metrics) to continue.")
    st.stop()

metrics = safe_read_excel(metrics_file)
required_cols = {"Player", "Unique one-liner (editable)", "Attended", "Attendance%", "H%", "M%", "L%", "Volatility%", "Risk Score", "PP Taken", "Archetype"}
missing = required_cols - set(metrics.columns)
if missing:
    st.error(f"Your metrics file is missing required columns: {sorted(list(missing))}")
    st.stop()

metrics["Player"] = metrics["Player"].astype(str).apply(clean_name)
metrics["player_key"] = metrics["Player"].apply(key_name)

# Force correct denominator display
metrics["Attended_out_of_24"] = metrics["Attended"].astype(int)
metrics["AttendancePct_out_of_24"] = (metrics["Attended_out_of_24"] / TOTAL_VALID_DROPS) * 100.0

# ==============================
# Master-derived enhancements (PP moments + streaks)
# ==============================
pp_details = {}
streaks = {}
valid_pp_second_drop = {}

if master_file is not None:
    raw_master = safe_read_excel(master_file, sheet="Summary", header=None)
    groups = parse_master_drop_groups(raw_master)
    q_lookup = {g["drop"]: g["question"] for g in groups}

    # Names in column C (index 2), rows 4..51 => 3..50
    player_to_row = {}
    for r in range(3, 51):
        nm = raw_master.iloc[r, 2] if raw_master.shape[1] > 2 else None
        if is_blank(nm):
            continue
        player_to_row[key_name(nm)] = r

    for _, row in metrics.iterrows():
        pk = row["player_key"]
        r = player_to_row.get(pk, None)
        if r is None:
            continue

        answered_flags = []
        pp_yes_drops = []
        pp_bucket_by_drop = {}

        for g in groups:
            resp = raw_master.iloc[r, g["resp_col"]]
            answered = not is_blank(resp)
            answered_flags.append(answered)

            b = raw_master.iloc[r, g["bucket_col"]] if g["bucket_col"] is not None else ""
            b = b.strip() if isinstance(b, str) else ""

            if g["pp_col"] is not None:
                pp = raw_master.iloc[r, g["pp_col"]]
                if isinstance(pp, str) and pp.strip().lower() == "yes":
                    pp_yes_drops.append(g["drop"])
                    if answered and b in ["H", "M", "L"]:
                        pp_bucket_by_drop[g["drop"]] = b

        longest, current = compute_streaks(answered_flags)
        streaks[pk] = {"longest": longest, "current": current}

        # Only first 2 count
        pp_total_marked = len(pp_yes_drops)
        pp_valid = pp_yes_drops[:2]

        moments = []
        if len(pp_valid) >= 1:
            d1 = int(pp_valid[0])
            moments.append({"drop": d1, "question": q_lookup.get(d1, f"Q{d1}"), "bucket": pp_bucket_by_drop.get(d1, ""), "ordinal": "First"})
        if len(pp_valid) >= 2:
            d2 = int(pp_valid[1])
            moments.append({"drop": d2, "question": q_lookup.get(d2, f"Q{d2}"), "bucket": pp_bucket_by_drop.get(d2, ""), "ordinal": "Second"})
            valid_pp_second_drop[pk] = d2

        pp_details[pk] = {"pp_total_marked": pp_total_marked, "pp_valid_count": len(pp_valid), "moments": moments}

metrics["CurrentStreak"] = metrics["player_key"].map(lambda k: streaks.get(k, {}).get("current", np.nan))
metrics["LongestStreak"] = metrics["player_key"].map(lambda k: streaks.get(k, {}).get("longest", np.nan))
metrics["Second_PP_by_Q"] = metrics["player_key"].map(lambda k: valid_pp_second_drop.get(k, np.nan))

# ==============================
# Section 1 — The Room
# ==============================
st.markdown("## Section 1 — The Room")
st.markdown('<div class="rpl-muted">Behavioral “shape” of the room based on H/M/L buckets, volatility, and power play usage (Drop 12 excluded).</div>', unsafe_allow_html=True)

colA, colB, colC, colD = st.columns(4)
n_players = metrics["Player"].nunique()
active_25 = int((metrics["AttendancePct_out_of_24"] >= 25).sum())
total_responses = int(metrics["Attended_out_of_24"].sum())
perfect = int((metrics["Attended_out_of_24"] == TOTAL_VALID_DROPS).sum())
pp_exhausted = int((metrics["PP Taken"].astype(int) >= 2).sum())

with colA:
    st.metric("Players", n_players)
with colB:
    st.metric("Active (≥25% attendance)", active_25)
with colC:
    st.metric("Total valid drops", TOTAL_VALID_DROPS)
with colD:
    st.metric("Total responses", total_responses)

colE, colF, colG, colH = st.columns(4)
with colE:
    st.metric("Perfect Attendance Club", perfect)
with colF:
    if metrics["CurrentStreak"].notna().any():
        streak10 = int((metrics["CurrentStreak"] >= 10).sum())
        st.metric("10+ Current Streak Club", streak10)
    else:
        st.metric("10+ Current Streak Club", "—")
with colG:
    st.metric("PP exhausted (≥2 marked)", pp_exhausted)
with colH:
    total_bucketed = total_responses if total_responses else 1
    room_H = float((metrics["H%"] / 100.0 * metrics["Attended_out_of_24"]).sum() / total_bucketed * 100.0)
    st.metric("Room consensus tilt (H share)", f"{room_H:.1f}%")

H_overall = float((metrics["H%"] / 100.0 * metrics["Attended_out_of_24"]).sum())
M_overall = float((metrics["M%"] / 100.0 * metrics["Attended_out_of_24"]).sum())
L_overall = float((metrics["L%"] / 100.0 * metrics["Attended_out_of_24"]).sum())

mix_df = pd.DataFrame({
    "Bucket": ["H", "M", "L"],
    "Count": [H_overall, M_overall, L_overall],
    "Meaning": [BUCKET_EXPLAIN["H"], BUCKET_EXPLAIN["M"], BUCKET_EXPLAIN["L"]]
})

c1, c2 = st.columns([1.2, 1.0])
with c1:
    fig = px.pie(mix_df, values="Count", names="Bucket", hover_data=["Meaning"], hole=0.55)
    fig = plotly_dark_defaults(fig)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(title="Overall H / M / L mix (weighted by participation)")
    st.plotly_chart(fig, use_container_width=True)
with c2:
    arch_counts = metrics["Archetype"].value_counts().reset_index()
    arch_counts.columns = ["Archetype", "Players"]
    fig2 = px.bar(arch_counts, x="Archetype", y="Players")
    fig2 = plotly_dark_defaults(fig2)
    fig2.update_layout(title="Archetype distribution")
    st.plotly_chart(fig2, use_container_width=True)

with st.expander("Metric explainer (tap to open)", expanded=False):
    st.markdown(
        """
- **H% / M% / L%**: Share of answered picks in the **Most popular / Middle / Least popular** bucket.
- **Risk Score**: Average bucket distance from consensus (**H=0, M=0.5, L=1**).
- **Volatility%**: How often your bucket changes between consecutive answered drops.
- **Attendance**: Number of answered valid drops out of **24**.
- **Power Play**: You can mark many, but only the **first two** count.
"""
    )
    st.markdown("### Archetypes (simple definitions)")
    for k in ["Strategist","Anchor","Wiseman","Maverick","Calibrator","Wildcard","Pragmatist","Ghost"]:
        st.markdown(f"**{k}** — {ARCH_DEF[k]}")

# ==============================
# Section 2 — Player Experience Centre
# ==============================
st.markdown("## Section 2 — Player Experience Centre")
st.markdown('<div class="rpl-muted">Pick a player to see: style, streaks, H/M/L mix, and Power Play moments.</div>', unsafe_allow_html=True)

players = metrics["Player"].tolist()
sel = st.selectbox("Select player", players, index=0)
p = metrics.loc[metrics["Player"] == sel].iloc[0]
pk = p["player_key"]

left, mid, right = st.columns([1.15, 1.0, 1.15])

with left:
    st.markdown('<div class="rpl-card">', unsafe_allow_html=True)
    st.markdown(f"### {p['Player']}")
    st.markdown(f"<span class='rpl-pill'>{p['Archetype']}</span>", unsafe_allow_html=True)
    st.markdown(f"<div class='rpl-small' style='margin-top:8px;'><b>Archetype logic:</b> {ARCH_DEF.get(p['Archetype'], '')}</div>", unsafe_allow_html=True)
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("**What's unique about you**")
    st.write(p["Unique one-liner (editable)"] if isinstance(p["Unique one-liner (editable)"], str) else "")
    st.markdown('</div>', unsafe_allow_html=True)

with mid:
    st.markdown('<div class="rpl-card">', unsafe_allow_html=True)
    st.markdown("### Your scoreboard")
    st.metric("Attendance (out of 24)", int(p["Attended_out_of_24"]))
    st.progress(min(max(float(p["AttendancePct_out_of_24"]) / 100.0, 0.0), 1.0))
    pp_used = int(p["PP Taken"]) if pd.notna(p["PP Taken"]) else 0
    st.markdown(f"<div class='rpl-muted'>Power Plays marked: <b>{pp_used}</b> / 2 count</div>", unsafe_allow_html=True)
    st.progress(min(pp_used / 2.0, 1.0))
    if pd.notna(p.get("CurrentStreak", np.nan)):
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.metric("Current streak", int(p["CurrentStreak"]))
        st.metric("Longest streak", int(p["LongestStreak"]))
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="rpl-card">', unsafe_allow_html=True)
    st.markdown("### Style signals")
    pmix = pd.DataFrame({
        "Bucket": ["H","M","L"],
        "Share": [float(p["H%"]), float(p["M%"]), float(p["L%"])],
        "Meaning": [BUCKET_EXPLAIN["H"], BUCKET_EXPLAIN["M"], BUCKET_EXPLAIN["L"]]
    })
    figp = px.pie(pmix, values="Share", names="Bucket", hover_data=["Meaning"], hole=0.55)
    figp = plotly_dark_defaults(figp)
    figp.update_traces(textposition="inside", textinfo="percent+label")
    figp.update_layout(title="Your H / M / L mix")
    st.plotly_chart(figp, use_container_width=True)
    st.markdown(f"<div class='rpl-small'><b>Risk Score:</b> {float(p['Risk Score']):.2f} &nbsp; • &nbsp; <b>Volatility:</b> {float(p['Volatility%']):.1f}%</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("### Power Play moments")
st.markdown('<div class="rpl-muted">Only the first two Power Plays count. If you marked more, your early conviction moments are the ones that matter.</div>', unsafe_allow_html=True)

if master_file is None:
    st.info("Upload the master file to show your Power Play questions + streaks.")
else:
    info = pp_details.get(pk, {"pp_total_marked": 0, "pp_valid_count": 0, "moments": []})
    moments = info.get("moments", [])

    if not moments:
        st.markdown('<div class="rpl-card">No Power Play moments recorded yet.</div>', unsafe_allow_html=True)
    else:
        box = '<div class="rpl-card">'
        for m in moments:
            qn = m["drop"]
            qtext = m["question"]
            b = m.get("bucket","")
            lbl = bucket_label(b)
            extra = f"— you took a <b>{lbl}</b> on this one." if lbl else ""
            box += f"<div style='margin-bottom:10px;'><b>{m['ordinal']} Power Play</b> on <b>Q{qn}</b>:<br/>{qtext}<br/><span class='rpl-muted'>{extra}</span></div>"
        if info.get("pp_total_marked", 0) > 2:
            box += f"<div class='rpl-small'>You marked Power Play <b>{info['pp_total_marked']}</b> times — only the first two count.</div>"
        box += "</div>"
        st.markdown(box, unsafe_allow_html=True)

# ==============================
# Section 3 — Risk Map + PP + Story Mode
# ==============================
st.markdown("## Section 3 — Risk + Power Plays + Story Mode")

st.markdown("### Risk vs Attendance Map")
st.markdown('<div class="rpl-muted">Y-axis is Risk Score (H→L). X-axis is attendance%. Hover shows only name + archetype.</div>', unsafe_allow_html=True)

map_df = metrics.copy()
map_df["Hover"] = map_df.apply(lambda r: f"{r['Player']}; {r['Archetype']}", axis=1)
figm = px.scatter(
    map_df,
    x="AttendancePct_out_of_24",
    y="Risk Score",
    color="Archetype",
    hover_name="Hover",
    hover_data={"AttendancePct_out_of_24": False, "Risk Score": False, "Archetype": False, "Player": False, "Hover": False},
    size="Attended_out_of_24",
    size_max=22
)
figm = plotly_dark_defaults(figm)
figm.update_layout(xaxis_title="Attendance (%)", yaxis_title="Risk Score (H→L)")
st.plotly_chart(figm, use_container_width=True)

st.markdown("### Power Play Status")
pp0 = metrics[metrics["PP Taken"].astype(int) == 0]["Player"].tolist()
pp1 = metrics[metrics["PP Taken"].astype(int) == 1]["Player"].tolist()
pp2p = metrics[metrics["PP Taken"].astype(int) >= 2]["Player"].tolist()

b1, b2, b3 = st.columns(3)
with b1:
    st.markdown('<div class="rpl-card">', unsafe_allow_html=True)
    st.markdown(f"**Conviction still loaded ({len(pp0)})**")
    st.write(", ".join(pp0) if pp0 else "—")
    st.markdown('</div>', unsafe_allow_html=True)
with b2:
    st.markdown('<div class="rpl-card">', unsafe_allow_html=True)
    st.markdown(f"**One-shot conviction ({len(pp1)})**")
    st.write(", ".join(pp1) if pp1 else "—")
    st.markdown('</div>', unsafe_allow_html=True)
with b3:
    st.markdown('<div class="rpl-card">', unsafe_allow_html=True)
    st.markdown(f"**Two chips spent ({len(pp2p)})**")
    st.write(", ".join(pp2p) if pp2p else "—")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("### Story Mode")
st.markdown('<div class="rpl-muted">A few patterns forming in the room — each card explains what the list means.</div>', unsafe_allow_html=True)

tabs = st.tabs(["Contrarian League", "Shape-Shifters", "Wildcards", "Conviction Still Loaded", "All-In Early"])

story_base = metrics.copy()
story_base = story_base[story_base["AttendancePct_out_of_24"] >= 30].copy()
story_base = story_base[story_base["Attended_out_of_24"] > 0].copy()

with tabs[0]:
    st.markdown('<div class="rpl-card">', unsafe_allow_html=True)
    st.markdown("**What this list means:** Players who most often choose the least popular option (high L%).")
    top = story_base.sort_values("L%", ascending=False).head(10)
    st.write(", ".join(top["Player"].tolist()) if len(top) else "—")
    st.markdown('</div>', unsafe_allow_html=True)

with tabs[1]:
    st.markdown('<div class="rpl-card">', unsafe_allow_html=True)
    st.markdown("**What this list means:** Players who change buckets frequently (high volatility) — adapts stance question by question.")
    top = story_base.sort_values("Volatility%", ascending=False).head(10)
    st.write(", ".join(top["Player"].tolist()) if len(top) else "—")
    st.markdown('</div>', unsafe_allow_html=True)

with tabs[2]:
    st.markdown('<div class="rpl-card">', unsafe_allow_html=True)
    st.markdown("**What this list means:** Wildcards are high-variance profiles (risk + volatility). Different from Contrarians: contrarians stay L-leaning; wildcards can swing anywhere.")
    top = story_base[story_base["Archetype"] == "Wildcard"].sort_values(["AttendancePct_out_of_24","Risk Score"], ascending=False).head(12)
    st.write(", ".join(top["Player"].tolist()) if len(top) else "—")
    st.markdown('</div>', unsafe_allow_html=True)

with tabs[3]:
    st.markdown('<div class="rpl-card">', unsafe_allow_html=True)
    st.markdown("**What this list means:** Players who haven’t used a Power Play yet — their conviction chips are fully in reserve.")
    top = story_base[story_base["PP Taken"].astype(int) == 0].sort_values("AttendancePct_out_of_24", ascending=False).head(20)
    st.write(", ".join(top["Player"].tolist()) if len(top) else "—")
    st.markdown('</div>', unsafe_allow_html=True)

with tabs[4]:
    st.markdown('<div class="rpl-card">', unsafe_allow_html=True)
    st.markdown("**What this list means:** Players who exhausted both valid Power Plays early (master-derived). We show the question number by which their second PP was spent.")
    if master_file is None or metrics["Second_PP_by_Q"].isna().all():
        st.write("Upload the master file to compute this list.")
    else:
        top = metrics.dropna(subset=["Second_PP_by_Q"]).sort_values("Second_PP_by_Q", ascending=True).head(12)
        items = [f"{r.Player} (by Q{int(r.Second_PP_by_Q)})" for r in top.itertuples()]
        st.write(", ".join(items) if items else "—")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("### Perfect Attendance Club (24/24)")
club = metrics[metrics["Attended_out_of_24"] == TOTAL_VALID_DROPS]["Player"].tolist()
st.markdown('<div class="rpl-card">', unsafe_allow_html=True)
st.write(", ".join(club) if club else "—")
st.markdown('</div>', unsafe_allow_html=True)

if show_data:
    st.markdown("## Debug: Data tables")
    st.dataframe(metrics.drop(columns=["player_key"]), use_container_width=True)
