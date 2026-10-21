"""Landslide susceptibility, Hamirpur and south Kangra: a read-only web app.

Everything shown here is loaded from files the notebook already saved
(study_area/Results, study_area/Bhuvan_inventory). Nothing is recomputed except
the "Try the model" tab, which runs the saved SVM on one point at a time.

Run locally from the repo root:  streamlit run app/streamlit_app.py
"""
import base64
import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "study_area" / "Results"
INVENTORY = ROOT / "study_area" / "Bhuvan_inventory"
ASSETS = Path(__file__).resolve().parent / "assets"

ZONE_ORDER = ["Very Low", "Low", "Moderate", "High", "Very High"]
ZONE_COLOURS = ["#1a9641", "#a6d96a", "#ffffbf", "#fdae61", "#d7191c"]
REPO_URL = "https://github.com/Subham1709/BTP"

st.set_page_config(page_title="Landslide susceptibility, Hamirpur", page_icon="⛰️", layout="wide")


# ---------- data loading (cached so each file is read once per server) ----------
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


@st.cache_data
def load_overlay():
    """Coloured zone PNG (made by app/make_overlay.py) as a data URI, plus its corners."""
    png = (ASSETS / "susceptibility_zones.png").read_bytes()
    bounds = json.loads((ASSETS / "susceptibility_zones_bounds.json").read_text())
    return "data:image/png;base64," + base64.b64encode(png).decode(), bounds


@st.cache_data
def load_districts():
    return json.loads((RESULTS / "hp_districts.geojson").read_text())


@st.cache_resource
def load_model():
    # dict saved by notebook Cell 13: {"model": sklearn Pipeline, "features": [...], "name": "SVM"}
    return joblib.load(RESULTS / "best_model.joblib")


def add_aspect_terms(df):
    """Aspect is circular, so the model takes cos/sin of it (same as the notebook)."""
    df = df.copy()
    df["northness"] = np.cos(np.radians(df["aspect"]))
    df["eastness"] = np.sin(np.radians(df["aspect"]))
    return df


models = load_csv(RESULTS / "model_comparison.csv")
tuning = load_csv(RESULTS / "tuning_default_vs_tuned.csv")
zones = load_csv(RESULTS / "susceptibility_zone_stats.csv")
exposure = load_csv(RESULTS / "risk_exposure_by_zone.csv")
districts = load_csv(RESULTS / "risk_by_district.csv")
roads = load_csv(RESULTS / "priority_road_stretches.csv")
importance = load_csv(RESULTS / "factor_importance.csv").rename(columns={"Unnamed: 0": "Factor"})
points = load_csv(INVENTORY / "ml_dataset_AOI.csv")

svm_spatial = tuning.query("Model == 'SVM' and Version == 'Tuned' and Test == 'Spatial 5 km CV'").iloc[0]
high_vh = zones[zones.Zone.isin(["High", "Very High"])]
exp_high_vh = exposure[exposure.Zone.isin(["High", "Very High"])]

# ---------- header ----------
st.title("Where are landslides most likely in Hamirpur and south Kangra?")
st.markdown(
    "B.Tech project, IIT Jodhpur (2026) · Subham Agrawal · "
    f"[Code and notebook on GitHub]({REPO_URL})  \n"
    "Satellite factors from Google Earth Engine, 4,431 mapped landslides from NRSC Bhuvan, "
    "five ML models tested on areas they had never seen, and a 30 m susceptibility map "
    "of about 2,360 km² in Himachal Pradesh."
)

c = st.columns(5)
c[0].metric("Mapped landslides", f"{4431:,}")
c[1].metric("Final model", "Tuned SVM")
c[2].metric("Spatial-CV ROC-AUC", f"{svm_spatial.ROC_AUC:.3f}")
c[3].metric("Area High or Very High", f"{high_vh['Area_%'].sum():.1f}%")
c[4].metric("Major roads in High + VH", f"{exp_high_vh.Road_km.sum():.1f} km")

tab_map, tab_models, tab_risk, tab_factors, tab_try = st.tabs(
    ["Map", "Model testing", "Zones and exposure", "Which factors matter", "Try the model"]
)

# ---------- 1. interactive map ----------
with tab_map:
    image, b = load_overlay()
    show_zones = st.checkbox("Susceptibility zones", value=True)
    show_roads = st.checkbox("40 priority road stretches", value=True)
    show_slides = st.checkbox("Mapped landslide points", value=False)

    layers = [
        pdk.Layer(
            "GeoJsonLayer", load_districts(), stroked=True, filled=False,
            get_line_color=[60, 60, 60], line_width_min_pixels=1,
        )
    ]
    if show_zones:
        layers.append(pdk.Layer(
            "BitmapLayer", image=pdk.types.String(image),  # String() stops pydeck reading the data URI as code
            bounds=[[b["west"], b["south"]], [b["west"], b["north"]],
                    [b["east"], b["north"]], [b["east"], b["south"]]],
        ))
    if show_slides:
        slides = points[points.label == 1][["lon", "lat", "season"]]
        layers.append(pdk.Layer(
            "ScatterplotLayer", slides, get_position=["lon", "lat"],
            get_fill_color=[90, 40, 140, 160], get_radius=60, pickable=True,
        ))
    if show_roads:
        layers.append(pdk.Layer(
            "ScatterplotLayer", roads, get_position=["mid_lon", "mid_lat"],
            get_fill_color=[20, 20, 20, 230], get_line_color=[255, 255, 255],
            stroked=True, line_width_min_pixels=1, radius_min_pixels=5, get_radius=250,
            pickable=True,
        ))

    st.pydeck_chart(pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=31.79, longitude=76.52, zoom=9.3),
        map_style="light",
        tooltip={"html": "{road_type} road · {length_m} m · {share_very_high_%}% in Very High"},
    ), height=620)

    legend = " ".join(
        f"<span style='background:{col};padding:2px 8px;border:1px solid #999'>{z}</span>"
        for z, col in zip(ZONE_ORDER, ZONE_COLOURS)
    )
    st.markdown(legend + " &nbsp; ⚫ priority road stretch (hover for details)", unsafe_allow_html=True)
    st.caption(
        "Zones are the tuned SVM's landslide probability on every 30 m pixel (2024 imagery), "
        "cut at 0.2 steps. Black dots mark the middle of road stretches that are mostly in Very High."
    )
    with st.expander("Report-quality map (600 dpi)"):
        st.image(str(RESULTS / "fig_susceptibility_final.png"))

# ---------- 2. model testing ----------
with tab_models:
    st.subheader("Random splits overstate accuracy")
    st.markdown(
        "Neighbouring points have nearly the same slope, rainfall and land cover, so a random "
        "train/test split lets a model memorise places. Each model was tested three ways. "
        "**Test B (whole 5 km blocks held out) is the one that matters**, because the map is used "
        "on ground with no recorded landslides."
    )
    metric = st.radio("Metric", ["ROC_AUC", "OA", "Kappa", "F1"], horizontal=True)
    chart = alt.Chart(models).mark_bar().encode(
        x=alt.X("Model:N", title=None),
        xOffset="Test:N",
        y=alt.Y(f"{metric}:Q", scale=alt.Scale(zero=False, domain=[0.3 if metric == "Kappa" else 0.6, 1])),
        color=alt.Color("Test:N", legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Model", "Test", alt.Tooltip(f"{metric}:Q", format=".3f")],
    ).properties(height=380)
    st.altair_chart(chart, width="stretch")
    st.markdown(
        "XGBoost is best on the random split (0.967) but last on spatial blocks (0.851): it was partly "
        "memorising locations. After tuning on spatial CV the SVM is best (0.905), so it is the final model. "
        "All tuned models are within 0.05 AUC of each other, so the testing design matters more than the algorithm."
    )
    st.markdown("**Default vs tuned settings**")
    st.dataframe(
        tuning.pivot_table(index="Model", columns=["Test", "Version"], values="ROC_AUC").round(3),
        width="stretch",
    )
    st.image(str(RESULTS / "roc_curves.png"), caption="ROC curves")

# ---------- 3. zones and exposure ----------
with tab_risk:
    left, right = st.columns(2)
    with left:
        st.subheader("Frequency ratio by zone")
        fr = alt.Chart(zones).mark_bar().encode(
            x=alt.X("Zone:N", sort=ZONE_ORDER, title=None),
            y=alt.Y("Frequency_ratio:Q", title="Frequency ratio"),
            color=alt.Color("Zone:N", sort=ZONE_ORDER, legend=None,
                            scale=alt.Scale(domain=ZONE_ORDER, range=ZONE_COLOURS)),
            tooltip=["Zone", alt.Tooltip("Area_%:Q", format=".1f"),
                     alt.Tooltip("Landslides_%:Q", format=".1f"),
                     alt.Tooltip("Frequency_ratio:Q", format=".2f")],
        ).properties(height=300)
        st.altair_chart(fr, width="stretch")
        st.caption(
            "Frequency ratio = share of landslides in a zone / share of area in it. "
            "It rises from 0.06 (Very Low) to 6.45 (Very High). This uses the training landslides, "
            "so it is a consistency check, not an independent test."
        )
    with right:
        st.subheader("What sits in each zone")
        what = st.selectbox("Show", ["Road_%", "Population_%", "Built_%", "Area_%"])
        ex = alt.Chart(exposure).mark_bar().encode(
            x=alt.X("Zone:N", sort=ZONE_ORDER, title=None),
            y=alt.Y(f"{what}:Q", title=what.replace("_", " ")),
            color=alt.Color("Zone:N", sort=ZONE_ORDER, legend=None,
                            scale=alt.Scale(domain=ZONE_ORDER, range=ZONE_COLOURS)),
            tooltip=["Zone", "Road_km", "Population", "Built_up_km2"],
        ).properties(height=300)
        st.altair_chart(ex, width="stretch")
        st.caption(
            f"High + Very High hold {exp_high_vh.Road_km.sum():.1f} km of major roads and about "
            f"{exp_high_vh.Population.sum():,.0f} people (GHSL population)."
        )

    st.subheader("By district")
    st.dataframe(districts.round(1), width="stretch", hide_index=True)
    st.subheader("Priority road stretches for slope protection")
    st.caption(f"{len(roads)} stretches, {roads.length_m.sum() / 1000:.1f} km in total.")
    st.dataframe(roads.round(2), width="stretch", hide_index=True)

# ---------- 4. factor importance ----------
with tab_factors:
    st.subheader("Permutation importance")
    st.markdown(
        "How much ROC-AUC drops when one factor is shuffled (test-A split). Slope matters most for "
        "Random Forest, rainfall for XGBoost. Caveat: the 5.5 km CHIRPS rainfall grid has only a few "
        "distinct values here, so rainfall may partly act as a location label."
    )
    long = importance.melt("Factor", var_name="Model", value_name="AUC drop")
    imp = alt.Chart(long).mark_bar().encode(
        y=alt.Y("Factor:N", sort="-x", title=None),
        x=alt.X("AUC drop:Q"),
        color=alt.Color("Model:N", legend=alt.Legend(orient="bottom", title=None)),
        yOffset="Model:N",
        tooltip=["Factor", "Model", alt.Tooltip("AUC drop:Q", format=".3f")],
    ).properties(height=480)
    st.altair_chart(imp, width="stretch")

# ---------- 5. try the saved model ----------
with tab_try:
    st.subheader("Run the final SVM on one point")
    st.markdown(
        "Pick one of the 8,696 training points, then change its slope, rainfall or other factors "
        "to see how the predicted landslide probability moves."
    )
    bundle = load_model()
    pts = add_aspect_terms(points)

    kind = st.radio("Start from", ["A landslide point", "A non-landslide point"], horizontal=True)
    pool = pts[pts.label == (1 if kind == "A landslide point" else 0)].reset_index(drop=True)
    idx = st.slider("Point number", 0, len(pool) - 1, 0)
    row = pool.loc[idx]

    sliders = {
        "slope": ("Slope (°)", 0.0, 67.0),
        "mean_annual_rainfall_mm": ("Mean annual rainfall (mm)", 1550.0, 2500.0),
        "elevation": ("Elevation (m)", 450.0, 1560.0),
        "distance_to_road": ("Distance to road (m)", 0.0, 5000.0),
        "ndvi": ("NDVI (vegetation)", -0.17, 0.82),
        "aspect": ("Aspect (° from north)", 0.0, 359.0),
    }
    edited = row.copy()
    cols = st.columns(3)
    for i, (feat, (label, lo, hi)) in enumerate(sliders.items()):
        edited[feat] = cols[i % 3].slider(label, lo, hi, float(np.clip(row[feat], lo, hi)), key=f"{feat}_{kind}_{idx}")

    raw = [f for f in bundle["features"] if f not in ("northness", "eastness")] + ["aspect"]
    X = add_aspect_terms(pd.DataFrame([edited[raw]]).astype(float))[bundle["features"]]
    prob = bundle["model"].predict_proba(X.to_numpy())[0, 1]  # model was fitted on a plain array
    zone = ZONE_ORDER[min(int(prob // 0.2), 4)]

    a, b_, c_ = st.columns(3)
    a.metric("Predicted landslide probability", f"{prob:.2f}")
    b_.metric("Zone", zone)
    c_.metric("Actually a landslide?", "Yes" if row.label == 1 else "No", f"{int(row.season)} inventory" if row.label == 1 else None)
    st.caption(
        f"Location {row.lat:.4f}°N, {row.lon:.4f}°E. Other factors (curvature, TWI, drainage, "
        "distance to river, soil clay, NDWI, NDBI) stay at this point's values. "
        "Probabilities come from CalibratedClassifierCV on the SVM."
    )

st.divider()
st.caption(
    "Limitations: no lithology layer, 2023 dominates the inventory (~83% of slides), geotechnical "
    f"parameters not included. Full method in the [README and report]({REPO_URL})."
)
