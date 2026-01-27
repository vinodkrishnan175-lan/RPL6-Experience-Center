import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple

# -----------------------------
# App config
# -----------------------------
st.set_page_config(
    page_title="RPL 6 Predictor",
    page_icon="🏆",
    layout="wide",
)

st.title("🏆 RPL 6 Predictor")
st.caption("Behavioral tournament predictor — transparent inputs → assumptions → outputs (no hidden math).")


# -----------------------------
# Utilities
# -----------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def normalize_weights(weights: pd.Series) -> pd.Series:
    w = weights.astype(float).clip(lower=0.0)
    s = w.sum()
    if s <= 0:
        return pd.Series(np.ones(len(w)) / len(w), index=w.index)
    return w / s


def make_sample_participants() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"participant": "Aarav", "seed": 1, "skill": 0.70, "consistency": 0.60, "risk": 0.40},
            {"participant": "Diya", "seed": 2, "skill": 0.65, "consistency": 0.70, "risk": 0.35},
            {"participant": "Ishaan", "seed": 3, "skill": 0.62, "consistency": 0.55, "risk": 0.55},
            {"participant": "Meera", "seed": 4, "skill": 0.60, "consistency": 0.65, "risk": 0.30},
            {"participant": "Kabir", "seed": 5, "skill": 0.58, "consistency": 0.52, "risk": 0.60},
            {"participant": "Sara", "seed": 6, "skill": 0.56, "consistency": 0.60, "risk": 0.45},
            {"participant": "Vihaan", "seed": 7, "skill": 0.54, "consistency": 0.50, "risk": 0.55},
            {"participant": "Anaya", "seed": 8, "skill": 0.52, "consistency": 0.62, "risk": 0.35},
        ]
    )


def make_sample_questions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"question": "Q1", "weight": 1.0, "difficulty": 0.55, "variance": 0.40},
            {"question": "Q2", "weight": 1.0, "difficulty": 0.60, "variance": 0.45},
            {"question": "Q3", "weight": 1.0, "difficulty": 0.50, "variance": 0.35},
            {"question": "Q4", "weight": 1.0, "difficulty": 0.65, "variance": 0.55},
            {"question": "Q5", "weight": 1.0, "difficulty": 0.58, "variance": 0.42},
        ]
    )


def expected_question_score(skill: float, consistency: float, risk: float,
                            difficulty: float, variance: float,
                            base_points: float) -> float:
    """
    Transparent expected score model:
    - skill helps overcome difficulty
    - consistency reduces penalty from variance
    - risk can help on high-variance questions but hurts on low-variance ones
    """
    # Skill vs difficulty
    skill_term = (skill - difficulty)  # can be negative

    # Variance handling
    # Higher consistency -> less harmed by variance
    stability = (1.0 - variance) + consistency * variance  # in [~0,1]
    stability = clamp(stability, 0.0, 1.0)

    # Risk interaction:
    # - if variance high: risk slightly boosts
    # - if variance low: risk slightly penalizes
    risk_boost = (variance - 0.5) * (risk - 0.5)  # centered interaction

    # Compose (kept simple and explainable)
    raw = 0.5 + 0.9 * skill_term + 0.25 * risk_boost
    raw = clamp(raw, 0.0, 1.0)

    return base_points * raw * (0.6 + 0.4 * stability)


def build_matchups(participants: List[str], bracket_style: str) -> List[Tuple[str, str]]:
    """
    Build round-1 matchups.
    - "Seeded (1 vs N)" uses seed ordering.
    - "Random" shuffles.
    """
    names = participants.copy()
    if bracket_style == "Random":
        rng = np.random.default_rng(42)
        rng.shuffle(names)
        pairs = [(names[i], names[i + 1]) for i in range(0, len(names), 2)]
        return pairs
    # Seeded
    # assume list is already seed-sorted
    pairs = []
    n = len(names)
    for i in range(n // 2):
        pairs.append((names[i], names[n - 1 - i]))
    return pairs


def summarize_series_to_text(s: pd.Series, top_n=3) -> str:
    s2 = s.sort_values(ascending=False).head(top_n)
    return ", ".join([f"{idx} ({val:.2f})" for idx, val in s2.items()])


# -----------------------------
# Sidebar: data source & high-level controls
# -----------------------------
with st.sidebar:
    st.header("⚙️ Setup")

    data_mode = st.radio(
        "Choose data source",
        options=["Use built-in sample data", "Upload CSVs", "Manual entry (quick)"],
        index=0,
        help="You can start with sample data, then switch to CSV upload or manual entry.",
    )

    st.divider()

    st.subheader("Tournament controls")
    n_rounds = st.selectbox("Number of rounds", [1, 2, 3, 4], index=2)
    bracket_style = st.selectbox("Round-1 bracket", ["Seeded (1 vs N)", "Random"], index=0)

    st.divider()

    st.subheader("Scoring controls")
    base_points = st.number_input("Base points per question", min_value=1.0, max_value=100.0, value=10.0, step=1.0)
    pp_multiplier = st.number_input("Power Play multiplier", min_value=1.0, max_value=5.0, value=2.0, step=0.1)
    pp_per_round = st.number_input("Power Plays per participant per round", min_value=0, max_value=10, value=1, step=1)

    st.caption("Power Play is applied **per participant per round** to a **specific question**, and is fully shown in outputs.")


# -----------------------------
# Inputs section
# -----------------------------
st.markdown("## 1) Inputs")

participants_df = None
questions_df = None

colA, colB = st.columns(2, gap="large")

with colA:
    st.subheader("👥 Participants")
    st.caption("Required columns: participant, seed, skill, consistency, risk")

    if data_mode == "Use built-in sample data":
        participants_df = make_sample_participants()
        st.dataframe(participants_df, use_container_width=True)
        st.download_button(
            "Download sample participants CSV",
            data=participants_df.to_csv(index=False).encode("utf-8"),
            file_name="participants_sample.csv",
            mime="text/csv",
        )

    elif data_mode == "Upload CSVs":
        up_p = st.file_uploader("Upload participants CSV", type=["csv"], key="up_participants")
        if up_p is not None:
            participants_df = pd.read_csv(up_p)
            st.dataframe(participants_df, use_container_width=True)

    else:
        st.caption("Quick manual entry (edit in-table). For larger rosters, use CSV upload.")
        participants_df = st.data_editor(
            make_sample_participants(),
            num_rows="dynamic",
            use_container_width=True,
            key="participants_editor",
        )

with colB:
    st.subheader("❓ Questions")
    st.caption("Required columns: question, weight, difficulty, variance")

    if data_mode == "Use built-in sample data":
        questions_df = make_sample_questions()
        st.dataframe(questions_df, use_container_width=True)
        st.download_button(
            "Download sample questions CSV",
            data=questions_df.to_csv(index=False).encode("utf-8"),
            file_name="questions_sample.csv",
            mime="text/csv",
        )

    elif data_mode == "Upload CSVs":
        up_q = st.file_uploader("Upload questions CSV", type=["csv"], key="up_questions")
        if up_q is not None:
            questions_df = pd.read_csv(up_q)
            st.dataframe(questions_df, use_container_width=True)

    else:
        questions_df = st.data_editor(
            make_sample_questions(),
            num_rows="dynamic",
            use_container_width=True,
            key="questions_editor",
        )

st.markdown("---")


# Validate inputs
def validate_inputs(p: pd.DataFrame, q: pd.DataFrame) -> Tuple[bool, List[str]]:
    errors = []
    if p is None or q is None:
        errors.append("Please provide both participants and questions.")
        return False, errors

    req_p = ["participant", "seed", "skill", "consistency", "risk"]
    req_q = ["question", "weight", "difficulty", "variance"]

    for c in req_p:
        if c not in p.columns:
            errors.append(f"Participants missing column: '{c}'")
    for c in req_q:
        if c not in q.columns:
            errors.append(f"Questions missing column: '{c}'")

    if errors:
        return False, errors

    if p["participant"].isna().any() or (p["participant"].astype(str).str.strip() == "").any():
        errors.append("Participants: 'participant' names cannot be blank.")
    if p["participant"].duplicated().any():
        errors.append("Participants: duplicate participant names detected (must be unique).")

    try:
        n = len(p)
        if n < 2 or (n & (n - 1)) != 0:
            errors.append("Number of participants must be a power of 2 (e.g., 2, 4, 8, 16).")
    except Exception:
        errors.append("Participants table looks invalid.")

    try:
        if len(q) < 1:
            errors.append("You must have at least 1 question.")
    except Exception:
        errors.append("Questions table looks invalid.")

    return (len(errors) == 0), errors


ok, errs = validate_inputs(participants_df, questions_df)
if not ok:
    st.error("Input issues detected:")
    for e in errs:
        st.write(f"- {e}")
    st.stop()

# Clean and normalize
participants_df = participants_df.copy()
questions_df = questions_df.copy()

participants_df["participant"] = participants_df["participant"].astype(str).str.strip()
participants_df["seed"] = participants_df["seed"].apply(safe_int)
for c in ["skill", "consistency", "risk"]:
    participants_df[c] = participants_df[c].apply(safe_float).clip(0.0, 1.0)

questions_df["question"] = questions_df["question"].astype(str).str.strip()
questions_df["weight"] = questions_df["weight"].apply(safe_float).clip(0.0, None)
questions_df["difficulty"] = questions_df["difficulty"].apply(safe_float).clip(0.0, 1.0)
questions_df["variance"] = questions_df["variance"].apply(safe_float).clip(0.0, 1.0)

# Sort by seed
participants_df = participants_df.sort_values(["seed", "participant"]).reset_index(drop=True)

# -----------------------------
# Power Play input (explicit)
# -----------------------------
st.markdown("### 🎯 Power Plays (explicit per participant per round and question)")

st.caption(
    "This section ensures PP usage is **participant-level and question-specific**. "
    "Every PP applied is shown later in the Processing and Results tables."
)

participants = participants_df["participant"].tolist()
questions = questions_df["question"].tolist()

pp_mode = st.radio(
    "Power Play selection mode",
    options=["Auto (highest impact question)", "Manual (you choose per participant/round)"],
    index=0,
    horizontal=True,
)

# Build PP table: round, participant -> question (or blank)
pp_records = []
for r in range(1, n_rounds + 1):
    for name in participants:
        pp_records.append({"round": r, "participant": name, "pp_question": ""})

pp_table = pd.DataFrame(pp_records)

if pp_mode == "Manual (you choose per participant/round)":
    st.info("Pick the question each participant uses PP on for each round (leave blank to use no PP).")
    edited = st.data_editor(
        pp_table,
        use_container_width=True,
        disabled=["round", "participant"],
        column_config={
            "pp_question": st.column_config.SelectboxColumn(
                "PP on question",
                options=[""] + questions,
                help="Which question gets PP multiplier for this participant in this round?",
            )
        },
        key="pp_editor",
    )
    pp_table = edited.copy()
else:
    st.success("Auto mode: PP will be assigned to the question that yields the biggest expected gain for each participant per round.")

st.markdown("---")


# -----------------------------
# Processing / assumptions section
# -----------------------------
st.markdown("## 2) Processing / Assumptions")
st.caption("Everything here is shown explicitly: question weights, per-question expected scores, and PP effects (no silent steps).")

# Normalize question weights
questions_df["weight_norm"] = normalize_weights(questions_df["weight"])

col1, col2, col3 = st.columns([1.2, 1.0, 1.0], gap="large")

with col1:
    st.subheader("📌 Assumptions used")
    st.markdown(
        f"""
- **Expected score per question** = a transparent function of participant traits (skill/consistency/risk) and question properties (difficulty/variance).
- **Question weights** are normalized to sum to 1 (shown below).
- **Power Play (PP)** multiplies the score on **one chosen question** by **{pp_multiplier:.2f}×**.
- **PP limit** = {pp_per_round} PP per participant per round (we enforce this).
- Bracket style (Round 1) = **{bracket_style}**.
"""
    )

with col2:
    st.subheader("🧮 Question weights (normalized)")
    st.dataframe(
        questions_df[["question", "weight", "weight_norm", "difficulty", "variance"]].sort_values("question"),
        use_container_width=True,
    )
    st.caption("Weights shown as both raw and normalized. Normalized weights are used in totals.")

with col3:
    st.subheader("🧩 Participant traits")
    st.dataframe(
        participants_df[["participant", "seed", "skill", "consistency", "risk"]],
        use_container_width=True,
    )


# Compute expected base scores per participant x question
score_rows = []
for _, p in participants_df.iterrows():
    for _, q in questions_df.iterrows():
        exp_score = expected_question_score(
            skill=p["skill"],
            consistency=p["consistency"],
            risk=p["risk"],
            difficulty=q["difficulty"],
            variance=q["variance"],
            base_points=base_points,
        )
        score_rows.append(
            {
                "participant": p["participant"],
                "question": q["question"],
                "expected_base_points": exp_score,
                "weight_norm": q["weight_norm"],
                "expected_weighted_points": exp_score * q["weight_norm"],
            }
        )

base_matrix = pd.DataFrame(score_rows)

st.markdown("### 📈 Expected per-question scores (before PP)")
st.dataframe(
    base_matrix.pivot(index="participant", columns="question", values="expected_base_points").round(2),
    use_container_width=True,
)

# Determine PP assignments
# Enforce pp_per_round limit: our UI provides exactly one selection per row, so we treat that as <=1
# If pp_per_round == 0, ignore PP
pp_table_effective = pp_table.copy()
if pp_per_round <= 0:
    pp_table_effective["pp_question"] = ""

if pp_mode == "Auto (highest impact question)":
    # For each participant, pick question that maximizes (multiplied - base) in weighted space
    # Gain = base * (pp_multiplier - 1) * weight_norm
    gains = base_matrix.merge(
        questions_df[["question", "weight_norm"]],
        on="question",
        how="left",
        suffixes=("", "_q"),
    )
    gains["expected_gain_weighted"] = gains["expected_base_points"] * (pp_multiplier - 1.0) * gains["weight_norm"]
    best_q = gains.sort_values("expected_gain_weighted", ascending=False).groupby("participant").head(1)
    best_map = dict(zip(best_q["participant"], best_q["question"]))

    # Apply same auto choice each round (simple and explicit)
    pp_table_effective["pp_question"] = pp_table_effective["participant"].map(best_map).fillna("")

# If user attempts >pp_per_round in future expansions, we'd enforce here.
# In current table, max 1 selection per participant/round.

# Build a PP effect table per participant x round x question
pp_effect_rows = []
for _, row in pp_table_effective.iterrows():
    r = int(row["round"])
    name = row["participant"]
    chosen_q = row["pp_question"] if isinstance(row["pp_question"], str) else ""
    chosen_q = chosen_q.strip()
    for q in questions:
        pp_used = (chosen_q == q) and (chosen_q != "")
        pp_effect_rows.append(
            {
                "round": r,
                "participant": name,
                "question": q,
                "pp_used": pp_used,
                "multiplier": (pp_multiplier if pp_used else 1.0),
            }
        )

pp_effect = pd.DataFrame(pp_effect_rows)

st.markdown("### 🎯 Power Play application table (explicit)")
# Show in a compact pivot: each participant x round -> chosen question(s)
pp_chosen_view = (
    pp_table_effective.groupby(["round", "participant"])["pp_question"]
    .apply(lambda s: ", ".join([x for x in s.astype(str).tolist() if x.strip() != ""]) if len(s) else "")
    .reset_index()
    .rename(columns={"pp_question": "PP on question"})
)
st.dataframe(pp_chosen_view, use_container_width=True)

# Combine base scores with PP effects to compute total expected per round
combined = base_matrix.merge(pp_effect, on=["participant", "question"], how="left")
combined["multiplier"] = combined["multiplier"].fillna(1.0)

combined["expected_points_after_pp"] = combined["expected_base_points"] * combined["multiplier"]
combined["expected_weighted_after_pp"] = combined["expected_points_after_pp"] * combined["weight_norm"]

# Repeat totals per round (since base expectation is same; PP can differ per round)
round_totals = []
for r in range(1, n_rounds + 1):
    tmp = combined.merge(
        pp_effect[pp_effect["round"] == r][["participant", "question", "multiplier", "pp_used"]],
        on=["participant", "question"],
        how="left",
        suffixes=("", "_r"),
    )
    tmp["multiplier_r"] = tmp["multiplier_r"].fillna(1.0)
    tmp["pp_used_r"] = tmp["pp_used_r"].fillna(False)

    tmp["expected_points_after_pp"] = tmp["expected_base_points"] * tmp["multiplier_r"]
    tmp["expected_weighted_after_pp"] = tmp["expected_points_after_pp"] * tmp["weight_norm"]

    totals = tmp.groupby("participant", as_index=False).agg(
        expected_total_points=("expected_points_after_pp", "sum"),
        expected_total_weighted=("expected_weighted_after_pp", "sum"),
        pp_questions_used=("pp_used_r", "sum"),
    )
    totals["round"] = r
    round_totals.append(totals)

round_totals_df = pd.concat(round_totals, ignore_index=True)

st.markdown("### ✅ Round-level totals (explicit, includes PP usage count)")
st.dataframe(
    round_totals_df.sort_values(["round", "expected_total_weighted"], ascending=[True, False]).round(3),
    use_container_width=True,
)

st.markdown("---")


# -----------------------------
# Results / outputs section
# -----------------------------
st.markdown("## 3) Results / Outputs")
st.caption("Predictions shown as matchups + winner probabilities + transparent score breakdowns.")

# Build round 1 matchups
matchups_r1 = build_matchups(participants, bracket_style)

# Simple win-prob model using weighted totals (Round 1 assumed)
r1 = round_totals_df[round_totals_df["round"] == 1].set_index("participant")

def win_prob(a_score: float, b_score: float, temperature: float = 0.15) -> float:
    # logistic on normalized difference (temperature controls softness)
    diff = (a_score - b_score)
    return 1.0 / (1.0 + np.exp(-diff / max(temperature, 1e-6)))

match_rows = []
for a, b in matchups_r1:
    a_s = float(r1.loc[a, "expected_total_weighted"])
    b_s = float(r1.loc[b, "expected_total_weighted"])
    p_a = win_prob(a_s, b_s)
    winner = a if p_a >= 0.5 else b
    match_rows.append(
        {
            "round": 1,
            "match": f"{a} vs {b}",
            "A": a,
            "B": b,
            "A_expected_weighted": a_s,
            "B_expected_weighted": b_s,
            "P(A wins)": p_a,
            "Predicted winner": winner,
        }
    )

match_df = pd.DataFrame(match_rows)

colR1, colR2 = st.columns([1.2, 1.0], gap="large")

with colR1:
    st.subheader("🥊 Round 1 matchups (predicted)")
    show_df = match_df.copy()
    show_df["P(A wins)"] = (show_df["P(A wins)"] * 100).round(1).astype(str) + "%"
    st.dataframe(show_df, use_container_width=True)

with colR2:
    st.subheader("🏅 Leaderboard (Round 1 expected)")
    leaderboard = r1.reset_index()[["participant", "expected_total_weighted", "pp_questions_used"]].copy()
    leaderboard = leaderboard.sort_values("expected_total_weighted", ascending=False)
    leaderboard["rank"] = range(1, len(leaderboard) + 1)
    leaderboard = leaderboard[["rank", "participant", "expected_total_weighted", "pp_questions_used"]]
    st.dataframe(leaderboard.round(3), use_container_width=True)

st.markdown("### 🔍 Explain a participant (fully explicit breakdown)")

pick = st.selectbox("Choose a participant to inspect", options=participants, index=0)

# pick round too
pick_round = st.selectbox("Choose a round to inspect", options=list(range(1, n_rounds + 1)), index=0)

# Build participant breakdown for that round
pp_r = pp_effect[pp_effect["round"] == pick_round].copy()
bd = base_matrix[base_matrix["participant"] == pick].merge(
    pp_r[pp_r["participant"] == pick][["question", "multiplier", "pp_used"]],
    on="question",
    how="left",
)
bd["multiplier"] = bd["multiplier"].fillna(1.0)
bd["pp_used"] = bd["pp_used"].fillna(False)
bd["after_pp"] = bd["expected_base_points"] * bd["multiplier"]
bd["weighted_after_pp"] = bd["after_pp"] * bd["weight_norm"]

# Add visible "PP delta"
bd["pp_delta_points"] = bd["after_pp"] - bd["expected_base_points"]
bd["pp_delta_weighted"] = bd["weighted_after_pp"] - (bd["expected_weighted_points"])

bd_view = bd[
    [
        "question",
        "expected_base_points",
        "weight_norm",
        "expected_weighted_points",
        "pp_used",
        "multiplier",
        "after_pp",
        "weighted_after_pp",
        "pp_delta_points",
        "pp_delta_weighted",
    ]
].copy()

st.dataframe(bd_view.round(4), use_container_width=True)

tot_line = bd_view[["weighted_after_pp", "pp_delta_weighted"]].sum()
st.info(
    f"Total weighted score (after PP) = **{tot_line['weighted_after_pp']:.4f}** | "
    f"Total PP weighted lift = **{tot_line['pp_delta_weighted']:.4f}**"
)

st.markdown("### 📤 Export outputs")
export_bundle = {
    "participants_clean.csv": participants_df.to_csv(index=False),
    "questions_clean.csv": questions_df.to_csv(index=False),
    "expected_base_matrix.csv": base_matrix.to_csv(index=False),
    "power_play_table.csv": pp_table_effective.to_csv(index=False),
    "round_totals.csv": round_totals_df.to_csv(index=False),
    "round1_matchups.csv": match_df.to_csv(index=False),
}
export_choice = st.selectbox("Choose an output to download", list(export_bundle.keys()))
st.download_button(
    "Download selected output",
    data=export_bundle[export_choice].encode("utf-8"),
    file_name=export_choice,
    mime="text/csv",
)

st.markdown("---")

with st.expander("📚 What changed in this version (mapped to your feedback)"):
    st.markdown(
        """
- **Clear sectioning**: `1) Inputs` → `2) Processing / Assumptions` → `3) Results / Outputs`.
- **No silent calculations**: every important intermediate is shown (normalized weights, base matrix, PP application, round totals).
- **Participant-level PP logic is explicit**: PP is chosen **per participant, per round, for a specific question**, and displayed in:
  - PP application table
  - round totals (with PP usage count)
  - participant breakdown (showing PP delta per question)
- **No debug lines**: no debug prints, no “fixed” markers — only user-facing outputs.
- **Visible UI/UX differences**: new PP control section + explicit breakdown tables + export panel.
"""
    )

st.caption("Tip: Start with sample data → confirm the logic visually → then switch to CSV upload.")
