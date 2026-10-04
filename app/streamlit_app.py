"""
TESS Exoplanet Candidate Explorer

A Streamlit app for browsing the pipeline's ranked list of unconfirmed TESS
planet candidates. It only reads the small files in app/assets/, which are
built by scripts/build_app_assets.py.

Run locally from the project root:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

APP = Path(__file__).parent
ASSETS = APP / "assets"
ROOT = APP.parent
REPO_URL = "https://github.com/emaadhasan/tess-transit-pipeline"

st.set_page_config(page_title="TESS Exoplanet Candidate Explorer", page_icon="🪐", layout="wide")


# Data
@st.cache_data
def load_data():
    cand = pd.read_csv(ASSETS / "candidates.csv")
    reference = json.loads((ASSETS / "feature_reference.json").read_text())
    recovery = pd.read_csv(ASSETS / "recovery_by_depth.csv")

    cand["tic_id"] = cand["target"].str.replace("TIC ", "", regex=False).astype(int)
    cand["triage"] = pd.cut(cand["score_physics"], bins=[-0.01, 0.2, 0.8, 1.01],
                            labels=["Likely impostor", "Uncertain", "High priority"]).astype(str)
    for flag in ["too_big", "weaker_than_training"]:
        if flag in cand.columns:
            cand[flag] = cand[flag].astype(str).str.lower().eq("true")
        else:
            cand[flag] = False
    return cand, reference, recovery


cand, reference, recovery = load_data()

TRIAGE_COLOURS = {"High priority": "#2e9e5b", "Uncertain": "#d9a400", "Likely impostor": "#c44e52"}

FEATURE_INFO = {
    "shape_ratio": ("Transit shape", "Near 1 is U-shaped, like a planet. Low values are V-shaped, typical of binary stars."),
    "odd_even_sigma": ("Odd vs. even dips (σ)", "How different alternating dips are. High values suggest a binary found at half its period."),
    "duty_cycle": ("Duty cycle", "Fraction of the orbit spent in transit. Planets are usually below about 0.1."),
    "secondary_ratio": ("Secondary dip ratio", "Size of any dip halfway through the orbit, relative to the transit. Binaries can be large."),
    "red_noise_beta": ("Correlated noise", "1 means random noise. Much higher values point to a variable star."),
    "depth_chi2": ("Transit consistency", "Around 1 means the individual transits agree. Much higher means they don't."),
    "duration_ratio": ("Duration vs. expected", "Measured transit length divided by what a planet around this star should give."),
    "planet_radius_re": ("Implied size (Earth radii)", "Above about 22 (2 Jupiter radii) is too large for almost any planet."),
    "centroid_source_offset_px": ("Source offset (pixels)", "How far the dip's source seems to be from the target star. Near 0 is the target itself."),
}


def fmt(value, digits=2):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:,.{digits}f}"


def lean(feature, value):
    """Which class's typical range is this value closer to (in units of that class's spread)?"""
    if value is None or (isinstance(value, float) and math.isnan(value)) or feature not in reference:
        return "n/a"
    if feature == "planet_radius_re":
        # Judged by physics, not the training data: confirmed planets skew large
        # only because big planets are the easiest to confirm.
        return "🔴 too large" if value > 22 else "🟢 plausible"
    p, f = reference[feature]["planet"], reference[feature]["false_positive"]
    spread_p = max(p["q75"] - p["q25"], 1e-9)
    spread_f = max(f["q75"] - f["q25"], 1e-9)
    closer_to_planet = abs(value - p["median"]) / spread_p < abs(value - f["median"]) / spread_f
    return "🟢 planet" if closer_to_planet else "🔴 impostor"


# Layout
st.title("TESS Exoplanet Candidate Explorer")
st.caption("Unconfirmed planet candidates from NASA's TESS mission, ranked by an automated "
           f"detection pipeline and classifier. [Source code on GitHub]({REPO_URL})")

tab_overview, tab_explore, tab_perf = st.tabs(["Overview", "Explore candidates", "How well it works"])

# Overview
with tab_overview:
    st.subheader("What this is")
    st.markdown(
        "When a planet passes in front of its star, the star dims slightly for a few hours. "
        "This project searches NASA's TESS data for those small, repeating dips, then uses a "
        "classifier to judge whether each signal looks like a real planet or an impostor, "
        "usually two stars eclipsing each other.\n\n"
        "This app shows the pipeline's ranking of **unconfirmed TESS planet candidates**: signals "
        "that astronomers have flagged but not yet confirmed or ruled out."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Confirmed planets recovered", "90%")
    c1.caption("376 of 419, at a signal-to-noise ratio of 7 or more")
    c2.metric("Planet vs. impostor (ROC AUC)", "0.94")
    c2.caption("5-fold cross-validation")
    c3.metric("Candidates ranked", f"{len(cand)}")
    c3.caption("unconfirmed TESS planet candidates")
    c4.metric("Cloud run, 1,066 stars", "4.5 h")
    c4.caption("vs. 11 hours on a laptop")

    st.subheader("The triage")
    counts = cand["triage"].value_counts()
    t1, t2, t3 = st.columns(3)
    t1.metric("🟢 High priority (score above 0.8)", int(counts.get("High priority", 0)))
    t2.metric("🟡 Uncertain", int(counts.get("Uncertain", 0)))
    t3.metric("🔴 Likely impostor (score below 0.2)", int(counts.get("Likely impostor", 0)))
    st.info(
        "Scores are a **ranking, not true probabilities**. The model was trained on a balanced set "
        "of confirmed planets and false positives, so a score of 0.9 means 'looks very planet-like', "
        "not 'a 90% chance of being a planet'. These are rankings of existing candidates, not discoveries."
    )

    st.subheader("How it works")
    st.markdown(
        "1. **Download** each star's light curve (brightness over time) from NASA's MAST archive.\n"
        "2. **Clean and detrend** it, removing glitches and the star's slow brightness changes.\n"
        "3. **Search** for repeating dips with Box Least Squares, protecting the transits with a second detrending pass.\n"
        "4. **Measure** 19 features that separate planets from impostors, such as transit shape and alternating dip depths.\n"
        "5. **Classify** each signal with a random forest trained on 897 confirmed planets and false positives.\n"
        "6. **Run at scale** on AWS Batch: 32 containers processing stars in parallel."
    )
    diagram = ROOT / "docs" / "architecture.png"
    if diagram.exists():
        st.image(str(diagram), caption="The AWS setup used to process stars at scale", width="stretch")

# Explore candidates
with tab_explore:
    st.subheader("Filter the candidates")
    f1, f2, f3 = st.columns(3)
    with f1:
        triage_pick = st.multiselect("Triage", list(TRIAGE_COLOURS), default=list(TRIAGE_COLOURS))
        search = st.text_input("Search by TOI or TIC number", placeholder="e.g. 1011.01 or 114018671")
    with f2:
        radius_top = float(math.ceil(min(cand["planet_radius_re"].max(skipna=True), 40)))
        radius_range = st.slider("Planet size (Earth radii)", 0.0, radius_top, (0.0, radius_top))
        period_range = st.slider("Orbital period (days)", 0.0, 15.0, (0.0, 15.0))
    with f3:
        min_score = st.slider("Minimum planet score", 0.0, 1.0, 0.0, 0.05)
        hide_big = st.checkbox("Hide candidates too big to be planets", value=False)
        hide_weak = st.checkbox("Hide signals weaker than the training data", value=False)

    view = cand[cand["triage"].isin(triage_pick)
                & cand["score_physics"].ge(min_score)
                & cand["period_days"].between(*period_range)]
    view = view[view["planet_radius_re"].isna() | view["planet_radius_re"].between(*radius_range)]
    if hide_big:
        view = view[~view["too_big"]]
    if hide_weak:
        view = view[~view["weaker_than_training"]]
    if search.strip():
        q = search.strip()
        view = view[view["toi"].astype(str).str.contains(q, regex=False)
                    | view["tic_id"].astype(str).str.contains(q, regex=False)]
    view = view.sort_values("score_physics", ascending=False).reset_index(drop=True)

    st.caption(f"Showing {len(view)} of {len(cand)} candidates. Click a row to see its details below.")
    table = view[["toi", "target", "triage", "score_physics", "score_all",
                  "period_days", "planet_radius_re", "depth_ppm", "snr_red"]]
    selection = st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "toi": st.column_config.NumberColumn("TOI", format="%.2f"),
            "target": "Star",
            "triage": "Triage",
            "score_physics": st.column_config.ProgressColumn("Planet score", min_value=0, max_value=1, format="%.2f"),
            "score_all": st.column_config.NumberColumn("All-features score", format="%.2f"),
            "period_days": st.column_config.NumberColumn("Period (days)", format="%.3f"),
            "planet_radius_re": st.column_config.NumberColumn("Size (Earth radii)", format="%.1f"),
            "depth_ppm": st.column_config.NumberColumn("Depth (ppm)", format="%.0f"),
            "snr_red": st.column_config.NumberColumn("SNR", format="%.1f"),
        },
    )

    if view.empty:
        st.warning("No candidates match these filters.")
    else:
        rows = selection.selection.rows if selection and selection.selection else []
        star = view.iloc[rows[0]] if rows else view.iloc[0]
        if not rows:
            st.caption("No row selected, so the top-ranked match is shown.")

        st.divider()
        st.subheader(f"TOI {star['toi']:.2f}  ·  {star['target']}")
        st.markdown(f"**{star['triage']}**  ·  "
                    f"[Latest astronomer notes on ExoFOP ↗](https://exofop.ipac.caltech.edu/tess/target.php?id={star['tic_id']})")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Planet score", fmt(star["score_physics"]))
        m2.metric("All-features score", fmt(star["score_all"]))
        m3.metric("Period (days)", fmt(star["period_days"], 3))
        m4.metric("Size (Earth radii)", fmt(star["planet_radius_re"], 1))
        m5.metric("SNR", fmt(star["snr_red"], 1))

        if star["too_big"]:
            st.warning("Implied size is above about 2 Jupiter radii, larger than almost any known planet. "
                       "This could be a small star or brown dwarf instead.")
        if star["weaker_than_training"]:
            st.warning("This signal is weaker than 95% of the training examples, so its score is less reliable.")

        left, right = st.columns([11, 9])
        with left:
            plot = ASSETS / "plots" / f"{star['target'].replace(' ', '_')}.webp"
            if plot.exists():
                st.image(str(plot), width="stretch",
                         caption="Top: light curve. Middle: period search. Bottom: all transits folded together.")
            else:
                st.info("No diagnostic plot available for this star.")
        with right:
            st.markdown("**Why it scored this way**")
            rows_out = []
            for feat, (label, _) in FEATURE_INFO.items():
                if feat in star.index:
                    val = star[feat]
                    p = reference.get(feat, {}).get("planet", {})
                    f = reference.get(feat, {}).get("false_positive", {})
                    rows_out.append({
                        "Feature": label,
                        "This star": fmt(val),
                        "Planets": fmt(p.get("median")),
                        "Impostors": fmt(f.get("median")),
                        "Leans": lean(feat, val),
                    })
            st.dataframe(pd.DataFrame(rows_out), hide_index=True, width="stretch")
            st.caption("Planets and Impostors show the typical (median) value for each group in the training data.")
            with st.expander("What do these features mean?"):
                for feat, (label, desc) in FEATURE_INFO.items():
                    st.markdown(f"**{label}:** {desc}")

# How well it works
with tab_perf:
    st.subheader("Detection: how many known planets does the pipeline find?")
    st.markdown("Of 419 confirmed planets, the pipeline recovered the catalog period with a signal-to-noise "
                "ratio of at least 7 for **90%**. Shallow transits, from smaller planets, are the hardest:")
    rec = recovery.copy()
    rec["recovered_pct"] = rec["recovered"] * 100
    order = rec["depth_bin"].tolist()
    chart = (alt.Chart(rec)
             .mark_bar(color="#4c78a8")
             .encode(x=alt.X("depth_bin:N", sort=order, title="Transit depth (parts per million)",
                             axis=alt.Axis(labelAngle=0)),
                     y=alt.Y("recovered_pct:Q", title="Planets recovered (%)", scale=alt.Scale(domain=[0, 100])),
                     tooltip=[alt.Tooltip("depth_bin:N", title="Depth"),
                              alt.Tooltip("recovered_pct:Q", title="Recovered (%)", format=".0f"),
                              alt.Tooltip("planets:Q", title="Planets")])
             .properties(height=320))
    st.altair_chart(chart, width="stretch")

    st.subheader("Classification: telling planets from impostors")
    st.markdown(
        "- A random forest on 19 features reaches **0.94 ROC AUC** (5-fold cross-validation) at separating "
        "confirmed planets from false positives among recovered signals.\n"
        "- Without any signal-strength features, it still reaches **0.92**, so it relies mainly on physics "
        "(transit shape, timing, alternating dips) rather than the bias that confirmed planets tend to have strong signals.\n"
        "- The ranking above uses this **physics-only** score."
    )

    st.subheader("How the candidates scored")
    hist = (alt.Chart(cand)
            .mark_bar()
            .encode(x=alt.X("score_physics:Q", bin=alt.Bin(step=0.05), title="Planet score"),
                    y=alt.Y("count()", title="Candidates"),
                    color=alt.Color("triage:N", title="Triage",
                                    scale=alt.Scale(domain=list(TRIAGE_COLOURS), range=list(TRIAGE_COLOURS.values()))))
            .properties(height=300))
    st.altair_chart(hist, width="stretch")

    st.subheader("Known limitations")
    st.markdown(
        "- **Brown dwarfs and small stars** are about the size of Jupiter, so their transits can look exactly like a "
        "hot Jupiter's. Only a mass measurement can tell them apart, so deep, high-scoring candidates deserve caution.\n"
        "- **Grazing planets**, which only clip the edge of their star, make V-shaped dips and can score low.\n"
        "- **Very variable stars** can hide a planet's signal.\n"
        "- Only candidates whose catalog period the pipeline recovered are ranked: about two-thirds of the "
        "eligible candidates, mostly the clearer signals."
    )
