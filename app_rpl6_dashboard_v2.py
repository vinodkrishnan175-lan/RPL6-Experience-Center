import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="RPL 6 — Behavioral Dashboard", layout="wide")

# -----------------------------
# Premium-ish styling (lightweight)
# -----------------------------
st.markdown(
    """
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
div[data-testid="stMetricLabel"] {opacity: 0.82;}
button[kind="secondary"] {border-radius: 999px !important; padding: 0.05rem 0.45rem !important; min-height: 0 !important;}
</style>
""",
    unsafe_allow_html=True,
)

def load_data(uploaded_file):
    if uploaded_file is None:
        return None
    xls = pd.ExcelFile(uploaded_file)
    sheet = "Player Metrics" if "Player Metrics" in xls.sheet_names else xls.sheet_names[0]
    return pd.read_excel(uploaded_file, sheet_name=sheet)

def clean_player_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Player"] = df["Player"].astype(str)
    df["Player_Clean"] = df["Player"].str.strip().str.replace(r"\s+", " ", regex=True)
    return df

def volatility_band(vol: float) -> str:
    if pd.isna(vol):
        return "—"
    if vol < 30:
        return "Low"
    if vol <= 60:
        return "Medium"
    return "High"

ARCH_DEF = {
    "Strategist": "Plays the long game: waits for clarity and commits selectively rather than rushing.",
    "Anchor": "Stays steady: consistent picks with low swing even when the room gets noisy.",
    "Maverick": "Trusts their own read: comfortable going against consensus when it feels right.",
    "Wildcard": "Leans into upside: embraces uncertainty and high-variance calls when opportunity shows up.",
    "Calibrator": "Adjusts actively: changes stance as signals and the room evolve.",
    "Wiseman": "Rides collective clarity: aligns with strong consensus and reinforces it with conviction.",
    "Pragmatist": "Mixes styles situationally: adapts approach question-by-question without dogma.",
    "Ghost": "Not enough signal yet: participation so far is too low to lock a clear style."
}

WHY_ARCH = {
    "Strategist": "High attendance + selective conviction behavior → a long-game approach.",
    "Anchor": "Low volatility + steady bucket mix → consistent style.",
    "Maverick": "Higher contrarian leaning (L%) → comfortable against consensus.",
    "Wildcard": "Higher risk posture + willingness to lean away from consensus → upside-seeking style.",
    "Calibrator": "Higher volatility → adjusts style frequently.",
    "Wiseman": "High H% + low L% → strong consensus alignment.",
    "Pragmatist": "Mixed H/M/L without a strong single leaning → situational approach.",
    "Ghost": "Low participation so far → not enough signal yet."
}

TOOLTIPS = {
    "H%": "% of your picks that matched the most popular option (crowd-aligned).",
    "M%": "% of picks in the middle — neither most popular nor least popular.",
    "L%": "% of your picks that were least popular (contrarian picks).",
    "Volatility": "How often your H/M/L bucket changes between answered drops. Higher = you switch style more often.",
    "AvgRisk": "Risk score from buckets: H=0, M=0.5, L=1. Higher = more contrarian on average."
}

def pop_help(label: str, body: str):
    with st.popover("?", help=f"Explain {label}"):
        st.write(body)

def psych_read(pp_used_total: int, b1: str, b2: str) -> str:
    if pp_used_total == 0:
        return "You’re holding conviction for a late swing — patience can be a weapon when reveal days begin."
    buckets = [b for b in [b1, b2] if isinstance(b, str) and b]
    if "L" in buckets:
        return "You use conviction to back instinct over consensus — bold minority calls can swing ranks fast on reveal days."
    if "H" in buckets:
        return "You use conviction to amplify collective clarity — if the crowd read holds, you stack points efficiently."
    if "M" in buckets:
        return "Your conviction sits in measured-confidence territory — steady accumulation without relying on chaos."
    return "Your conviction moments add a distinctive layer — reveal days will amplify what you backed."

# -----------------------------
# Header / Welcome banner
# -----------------------------
st.markdown(
    """
<div style="
    border-radius: 18px;
    padding: 18px 18px 14px 18px;
    background: linear-gradient(90deg, rgba(20,20,20,0.92), rgba(20,20,20,0.78));
    color: white;
    box-shadow: 0 8px 24px rgba(0,0,0,0.18);
    margin-bottom: 14px;">
  <div style="font-size: 24px; font-weight: 700; letter-spacing: 0.2px;">RPL 6 — Behavioral Dashboard</div>
  <div style="opacity: 0.88; margin-top: 4px; font-size: 14px;">
    A premium mirror of how you’ve been playing — conviction, consistency, and crowd distance (without the clutter).
  </div>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload latest analytics Excel", type=["xlsx"], accept_multiple_files=False)
    st.caption("Expected: RPL6_Analytics_Refreshed_Drop24.xlsx (sheet: Player Metrics).")
    show_debug = st.toggle("Show sanity-check card (temporary)", value=False)

df = load_data(uploaded)
if df is None:
    st.info("Upload the Excel file to render the dashboard.")
    st.stop()

df = clean_player_key(df)

for col in ["Archetype", "Archetype Definition", "Power Play moments", "AttendancePct", "Attended", "TotalDrops",
            "H%", "M%", "L%", "Volatility%", "AvgRisk", "PP_used_total", "PP_Q1", "PP_Q2", "PP_b1", "PP_b2"]:
    if col not in df.columns:
        df[col] = np.nan

df["Archetype Definition"] = df["Archetype Definition"].fillna(df["Archetype"].map(ARCH_DEF))

for c in ["AttendancePct","Attended","TotalDrops","H%","M%","L%","Volatility%","AvgRisk","PP_used_total","PP_Q1","PP_Q2"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Robust TotalDrops for perfect attendance
if df["TotalDrops"].notna().any():
    try:
        total_drops_global = int(df["TotalDrops"].mode(dropna=True).iloc[0])
    except Exception:
        total_drops_global = int(df["TotalDrops"].dropna().iloc[0])
else:
    total_drops_global = 0
df["TotalDrops"] = df["TotalDrops"].fillna(total_drops_global)

# -----------------------------
# Global Player Selector
# -----------------------------
players_sorted = sorted(df["Player_Clean"].dropna().unique().tolist())
sel_player = st.selectbox("Select Player", players_sorted, index=0)
row = df.loc[df["Player_Clean"] == sel_player].iloc[0]

# -----------------------------
# Player Profile
# -----------------------------
st.subheader("Player Profile")
c1, c2 = st.columns([2.2, 1.8], gap="large")

with c1:
    st.markdown(f"### {row['Player_Clean']}")
    archetype = str(row["Archetype"]) if pd.notna(row["Archetype"]) else "—"
    st.markdown(f"**Archetype:** `{archetype}`")
    st.caption(row.get("Archetype Definition", "") if pd.notna(row.get("Archetype Definition", "")) else "")

    st.markdown("**Why this archetype**")
    st.write(WHY_ARCH.get(archetype, "Behavioral mix + participation signal → this archetype label."))

    attended = int(row["Attended"]) if pd.notna(row["Attended"]) else 0
    total_drops = int(row["TotalDrops"]) if pd.notna(row["TotalDrops"]) else total_drops_global

    st.markdown("**Attendance**")
    st.progress(attended / total_drops if total_drops else 0.0, text=f"{attended} / {total_drops} drops")

    pp_used_total = int(row["PP_used_total"]) if pd.notna(row["PP_used_total"]) else 0
    pp_used_capped = min(pp_used_total, 2)
    st.markdown("**Power Plays used**")
    st.progress(pp_used_capped / 2, text=f"{pp_used_capped} / 2 (only first 2 count)")

with c2:
    st.markdown("**Style Snapshot**")
    m1, m2, m3 = st.columns(3, gap="small")
    with m1:
        st.metric("H%", f"{row['H%']:.1f}%" if pd.notna(row["H%"]) else "—")
        pop_help("H%", TOOLTIPS["H%"])
    with m2:
        st.metric("M%", f"{row['M%']:.1f}%" if pd.notna(row["M%"]) else "—")
        pop_help("M%", TOOLTIPS["M%"])
    with m3:
        st.metric("L%", f"{row['L%']:.1f}%" if pd.notna(row["L%"]) else "—")
        pop_help("L%", TOOLTIPS["L%"])

    s1, s2 = st.columns(2, gap="small")
    with s1:
        vol = row["Volatility%"]
        st.metric("Volatility", f"{vol:.1f}% • {volatility_band(vol)}" if pd.notna(vol) else "—")
        pop_help("Volatility", TOOLTIPS["Volatility"])
    with s2:
        risk = row["AvgRisk"]
        st.metric("AvgRisk", f"{risk:.2f}" if pd.notna(risk) else "—")
        pop_help("AvgRisk", TOOLTIPS["AvgRisk"])

    st.markdown("**Power Play Moments**")
    ppm = row.get("Power Play moments", "")
    st.write(ppm if isinstance(ppm, str) and ppm.strip() else "No Power Play moments recorded yet.")
    pr = psych_read(pp_used_total, str(row.get("PP_b1","")), str(row.get("PP_b2","")))
    st.caption(f"**Psych read:** {pr}")

st.divider()

# -----------------------------
# Story Mode
# -----------------------------
st.subheader("Story Mode")

story_df = df.copy()
story_df["AttendancePct"] = pd.to_numeric(story_df["AttendancePct"], errors="coerce")

# Filter names for most story lists (but NOT perfect attendance club)
story_names_df = story_df[story_df["AttendancePct"].fillna(0) >= 30].copy()
story_names_df = story_names_df[~story_names_df["Player_Clean"].str.lower().isin(["sai","arvind"])]

total_responses = int(df["Attended"].fillna(0).sum())
consensus_responses = int(np.round((df["Attended"].fillna(0) * df["H%"].fillna(0) / 100).sum()))

# Perfect attendance club: robust definition, no filters, no caps
perfect = df.copy()
perfect = perfect[(perfect["Attended"] >= (perfect["TotalDrops"] - 0.001)) | (perfect["AttendancePct"].fillna(0) >= 99.9)]
perfect = perfect.sort_values("Player_Clean")

both_pp = story_names_df[story_names_df["PP_used_total"].fillna(0) >= 2].copy().sort_values("PP_Q2")
contrarian = story_names_df.sort_values("L%", ascending=False).head(5)
anchors = story_names_df.sort_values("Volatility%").head(5)
no_pp = story_names_df[story_names_df["PP_used_total"].fillna(0) == 0].sort_values("Player_Clean")

tabs = st.tabs(["Meta", "Contrarian League", "Anchors", "Conviction Used Early", "Conviction Still Loaded", "Perfect Attendance Club"])

with tabs[0]:
    st.markdown("### The Meta So Far")
    st.caption("What this means: how often the room picked the most popular option overall.")
    st.write(f"Out of **{total_responses}** total responses, **{consensus_responses}** were crowd-aligned (**H**) — that’s why the room is leaning consensus.")
with tabs[1]:
    st.markdown("### The Contrarian League")
    st.caption("What this means: the players most willing to pick the least-popular option (L).")
    st.dataframe(contrarian[["Player_Clean","Archetype","L%","AttendancePct"]].rename(columns={"Player_Clean":"Player"}), hide_index=True, use_container_width=True)
with tabs[2]:
    st.markdown("### The Anchors")
    st.caption("What this means: the steadiest styles — lowest switching between H/M/L over time.")
    st.dataframe(anchors[["Player_Clean","Archetype","Volatility%","AttendancePct"]].rename(columns={"Player_Clean":"Player"}), hide_index=True, use_container_width=True)
with tabs[3]:
    st.markdown("### Conviction Used Early")
    st.caption("What this means: players who exhausted BOTH Power Plays earliest — shows early high conviction.")
    if len(both_pp) == 0:
        st.write("No players have used both Power Plays yet (after filters).")
    else:
        show = both_pp.head(5).copy()
        show["Exhausted by"] = show["PP_Q2"].apply(lambda x: f"Q{int(x)}" if pd.notna(x) else "—")
        st.dataframe(show[["Player_Clean","Exhausted by","Archetype","AttendancePct"]].rename(columns={"Player_Clean":"Player"}), hide_index=True, use_container_width=True)
with tabs[4]:
    st.markdown("### Conviction Still Loaded")
    st.caption("What this means: players who haven’t used any Power Play yet — potential late-surge ammunition.")
    st.write(", ".join(no_pp["Player_Clean"].tolist()) if len(no_pp) else "—")
with tabs[5]:
    st.markdown("### Perfect Attendance Club")
    st.caption("What this means: players who have answered every valid drop so far (Drop 12 excluded).")
    st.write(", ".join(perfect["Player_Clean"].tolist()) if len(perfect) else "—")

st.divider()

# -----------------------------
# Risk vs Attendance Map
# -----------------------------
st.subheader("Risk vs Attendance Map")
st.caption("X = attendance %, Y = average contrarian-lean (0 = consensus-heavy, 1 = contrarian-heavy).")

plot_df = df.copy()
plot_df["AttendancePct"] = pd.to_numeric(plot_df["AttendancePct"], errors="coerce")
plot_df["AvgRisk"] = pd.to_numeric(plot_df["AvgRisk"], errors="coerce")

fig = px.scatter(
    plot_df,
    x="AttendancePct",
    y="AvgRisk",
    color="Archetype",
    hover_data={
        "Player_Clean": True,
        "Archetype": True,
        "AttendancePct": ':.1f',
        "AvgRisk": ':.2f',
        "H%": ':.1f',
        "M%": ':.1f',
        "L%": ':.1f',
    },
    labels={"AttendancePct":"Attendance %", "AvgRisk":"AvgRisk"},
)

fig.update_layout(xaxis=dict(range=[0, 100]), yaxis=dict(range=[0, 1]), margin=dict(l=10, r=10, t=10, b=10))
fig.add_annotation(x=85, y=0.12, text="Consensus Regulars", showarrow=False, opacity=0.6)
fig.add_annotation(x=85, y=0.88, text="Contrarian Regulars", showarrow=False, opacity=0.6)
fig.add_annotation(x=15, y=0.12, text="Consensus Sparsely-Seen", showarrow=False, opacity=0.6)
fig.add_annotation(x=15, y=0.88, text="Contrarian Sparsely-Seen", showarrow=False, opacity=0.6)

st.caption(f"Points plotted: **{int(plot_df['Player_Clean'].nunique())}** (should be 48)")
st.plotly_chart(fig, use_container_width=True)

st.divider()

# -----------------------------
# Power Play Status (3 boxes) + counts
# -----------------------------
st.subheader("Power Play Status")

exhausted_df = df[df["PP_used_total"].fillna(0) >= 2].copy().sort_values("PP_Q2")
once_df = df[df["PP_used_total"].fillna(0) == 1].copy().sort_values("PP_Q1")
none_df = df[df["PP_used_total"].fillna(0) == 0].copy().sort_values("Player_Clean")

b1, b2, b3 = st.columns(3, gap="large")

with b1:
    st.markdown(f"### Exhausted both Power Plays ({len(exhausted_df)})")
    if len(exhausted_df):
        exhausted_df["Exhausted by"] = exhausted_df["PP_Q2"].apply(lambda x: f"Q{int(x)}" if pd.notna(x) else "—")
        st.dataframe(exhausted_df[["Player_Clean","Exhausted by"]].rename(columns={"Player_Clean":"Player"}), hide_index=True, use_container_width=True)
    else:
        st.write("—")

with b2:
    st.markdown(f"### Used Power Play only once ({len(once_df)})")
    if len(once_df):
        once_df["Used on"] = once_df["PP_Q1"].apply(lambda x: f"Q{int(x)}" if pd.notna(x) else "—")
        st.dataframe(once_df[["Player_Clean","Used on"]].rename(columns={"Player_Clean":"Player"}), hide_index=True, use_container_width=True)
    else:
        st.write("—")

with b3:
    st.markdown(f"### No Power Play used ({len(none_df)})")
    st.write(", ".join(none_df["Player_Clean"].tolist()) if len(none_df) else "—")

if show_debug:
    st.divider()
    st.subheader("Sanity Check (temporary)")
    st.write(f"- Total players: **{int(df['Player_Clean'].nunique())}** (should be 48)")
    st.write(f"- Total drops (mode): **{total_drops_global}**")
    st.write(f"- Perfect attendance players (robust): **{len(perfect)}**")
    st.write(f"- Missing AttendancePct values: **{int(df['AttendancePct'].isna().sum())}**")
    st.write(f"- Missing AvgRisk values: **{int(df['AvgRisk'].isna().sum())}**")
