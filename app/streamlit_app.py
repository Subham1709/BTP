"""Landslide susceptibility, Hamirpur and south Kangra - the project told as a story.

Every chapter reads files the notebook already saved (study_area/Results,
study_area/Bhuvan_inventory). Nothing heavy is recomputed; only the last
"what if" chapter runs the saved SVM on one point at a time.

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
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "study_area" / "Results"
INVENTORY = ROOT / "study_area" / "Bhuvan_inventory"
ASSETS = Path(__file__).resolve().parent / "assets"
REPO_URL = "https://github.com/Subham1709/BTP"

ZONE_ORDER = ["Very Low", "Low", "Moderate", "High", "Very High"]
ZONE_COLOURS = ["#1a9641", "#a6d96a", "#ffffbf", "#fdae61", "#d7191c"]
MODEL_ORDER = ["kNN", "SVM", "ANN", "Random Forest", "XGBoost"]
NICE = {  # readable factor names for charts
    "slope": "Slope", "northness": "Northness (aspect)", "eastness": "Eastness (aspect)",
    "mean_annual_rainfall_mm": "Mean annual rainfall", "elevation": "Elevation", "ndbi": "NDBI (built-up)",
    "distance_to_road": "Distance to road", "ndwi": "NDWI (water)", "plan_curvature": "Curvature",
    "ndvi": "NDVI (vegetation)", "distance_to_river": "Distance to river", "soil_clay": "Soil clay",
    "drainage_density": "Drainage density", "twi": "Wetness index (TWI)",
}
UNITS = {"slope": "degrees", "elevation": "m", "mean_annual_rainfall_mm": "mm/year",
         "distance_to_road": "m", "distance_to_river": "m", "soil_clay": "%", "drainage_density": "km/km²"}

st.set_page_config(page_title="Landslide susceptibility, Hamirpur", page_icon="⛰️", layout="wide")


# ---------- data (cached, read once per server) ----------
@st.cache_data
def load_csv(path, **kw):
    return pd.read_csv(path, **kw)


@st.cache_data
def load_overlay():
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
    df = df.copy()
    df["northness"] = np.cos(np.radians(df["aspect"]))
    df["eastness"] = np.sin(np.radians(df["aspect"]))
    return df


models = load_csv(RESULTS / "model_comparison.csv")
tuning = load_csv(RESULTS / "tuning_default_vs_tuned.csv")
settings = load_csv(RESULTS / "tuning_best_settings.csv", index_col=0)
zones = load_csv(RESULTS / "susceptibility_zone_stats.csv")
exposure = load_csv(RESULTS / "risk_exposure_by_zone.csv")
districts = load_csv(RESULTS / "risk_by_district.csv")
roads = load_csv(RESULTS / "priority_road_stretches.csv")
importance = load_csv(RESULTS / "factor_importance.csv").rename(columns={"Unnamed: 0": "Factor"})
change = load_csv(RESULTS / "change_timeseries.csv")
veg = load_csv(RESULTS / "vegetation_change_vs_landslides.csv")
points = load_csv(INVENTORY / "ml_dataset_AOI.csv")
lith = load_csv(ASSETS / "inventory_lithology.csv")

svm_spatial = tuning.query("Model == 'SVM' and Version == 'Tuned' and Test == 'Spatial 5 km CV'").iloc[0]
hv = zones[zones.Zone.isin(["High", "Very High"])]
exp_hv = exposure[exposure.Zone.isin(["High", "Very High"])]
models["Short"] = models.Test.str[0]  # A, B, C


def auc(model, test):
    return float(models[(models.Model == model) & (models.Short == test)].ROC_AUC.iloc[0])


# ---------- chapter navigation ----------
CHAPTERS = [
    "The question",
    "1 · The problem",
    "2 · Choosing the study area",
    "3 · Building the database",
    "4 · The testing trap",
    "5 · Picking the model",
    "6 · What drives landslides",
    "7 · The susceptibility map",
    "8 · Who is exposed",
    "9 · What if? Try the model",
    "10 · Limits and conclusions",
]
TIMES = ["0:30", "1:00", "1:30", "2:00", "2:00", "1:00", "1:30", "1:30", "1:30", "1:00", "1:00"]

if "ch" not in st.session_state:
    st.session_state.ch = 0


def go(step):
    st.session_state.ch = int(np.clip(st.session_state.ch + step, 0, len(CHAPTERS) - 1))


with st.sidebar:
    st.markdown("### ⛰️ Landslide story")
    st.radio("Chapter", range(len(CHAPTERS)), format_func=lambda i: CHAPTERS[i], key="ch", label_visibility="collapsed")
    st.divider()
    notes_on = st.toggle("Speaker notes", value=False, help="Talking points and a time budget for each chapter.")
    st.caption(f"B.Tech project, IIT Jodhpur (2026) · Subham Agrawal · [GitHub]({REPO_URL})")

ch = st.session_state.ch
# jump back to the top whenever the chapter changes (Streamlit keeps the scroll position otherwise)
components.html(f"""<script>/* chapter {ch} */
const d = window.parent.document;
for (const sel of ['[data-testid="stMain"]', 'section.main', '[data-testid="stAppViewContainer"]']) {{
  const el = d.querySelector(sel); if (el) el.scrollTo({{top: 0}});
}}
window.parent.scrollTo(0, 0);
</script>""", height=0)


def notes(*lines):
    if notes_on:
        with st.expander(f"🎤 Speaker notes · about {TIMES[ch]} min", expanded=True):
            for line in lines:
                st.markdown(f"- {line}")


def nav():
    st.divider()
    a, mid, b = st.columns([1, 3, 1])
    if ch > 0:
        a.button("← Back", on_click=go, args=(-1,), width="stretch")
    if ch < len(CHAPTERS) - 1:
        b.button("Next →", on_click=go, args=(1,), type="primary", width="stretch")
        mid.markdown(f"<div style='text-align:right;color:grey;padding-top:8px'>Up next: {CHAPTERS[ch + 1]}</div>",
                     unsafe_allow_html=True)


def takeaway(text):
    st.success(f"**Takeaway:** {text}")


# =====================================================================
# 0. The question
# =====================================================================
if ch == 0:
    st.title("Where will the next landslide happen in Hamirpur and south Kangra?")
    st.markdown(
        "#### And which roads and people are standing on that ground?\n"
        "In this project I built a map that answers this for about **2,360 km²** of the Shivalik hills of "
        "Himachal Pradesh, using free satellite data, a government landslide inventory and machine learning. "
        "The story below follows the order in which I solved the problem."
    )
    c = st.columns(4)
    c[0].metric("Mapped landslides used", "4,431")
    c[1].metric("Conditioning factors", "13")
    c[2].metric("Models compared", "5")
    c[3].metric("Map resolution", "30 m")
    st.image(str(RESULTS / "fig_susceptibility_final.png"),
             caption="Where we end up: the final susceptibility map (tuned SVM, 2024 imagery).")
    notes(
        "One line on who I am and the title.",
        "Ask the question on screen: where is the next landslide most likely, and who is on that ground?",
        "Say: I'll tell it as the problems I hit and how I solved them, and the numbers will back each step.",
    )
    nav()

# =====================================================================
# 1. The problem
# =====================================================================
elif ch == 1:
    st.header("1 · The problem: the 2023 monsoon")
    left, right = st.columns([3, 2])
    with left:
        st.markdown(
            "In the 2023 monsoon, NRSC mapped **13,569 landslides** across Himachal Pradesh, against 923 in 2014. "
            "Roads were cut, houses were lost, and most of the damage came from slopes that nobody had flagged.\n\n"
            "A **susceptibility map** answers *where* landslides are likely, based on the terrain, rain, land cover "
            "and the places where they have happened before. It does not say *when*. It is the first thing a "
            "planner, a road engineer or a disaster cell needs."
        )
        st.markdown("**The objectives I was given:**")
        st.markdown(
            "1. Build a GIS database of terrain, geology, soil, climate, hydrology and land cover.\n"
            "2. Find and quantify the factors that control landslides.\n"
            "3. Make a susceptibility map from a 10-15 year inventory.\n"
            "4. Build and compare ML models (RF, SVM, XGBoost, ANN).\n"
            "5. Mark High / Very High zones and turn them into a risk framework."
        )
    with right:
        seasons = pd.DataFrame({"Season": ["2014", "2017", "2023"], "Landslides": [923, 172, 13569]})
        st.altair_chart(alt.Chart(seasons).mark_bar(color="#d7191c").encode(
            x=alt.X("Season:N", title="Monsoon season"), y=alt.Y("Landslides:Q", title="Landslides mapped in HP"),
            tooltip=["Season", "Landslides"]).properties(height=330, title="Bhuvan seasonal inventory, all of HP"),
            width="stretch")
    takeaway("landslides in HP are frequent and getting costly, and planners need to know where the risky ground is.")
    notes(
        "Point to the 2023 bar: 13,569 slides in one season.",
        "Define susceptibility = where, not when. (Forecasting is 'when'; that's a different problem.)",
        "Read the 5 objectives quickly: I'll come back and tick them off in the last chapter.",
    )
    nav()

# =====================================================================
# 2. Study area
# =====================================================================
elif ch == 2:
    st.header("2 · Choosing the study area: follow the data")
    st.markdown(
        "**Problem I hit:** my first area, Kinnaur, had too few mapped landslides to train a model. "
        "An earlier pipeline for Kedarnath had the same issue: with only a handful of landslide samples, its "
        "near-perfect accuracy meant nothing.\n\n"
        "**What I did:** I downloaded the full Bhuvan inventory for Himachal Pradesh and counted landslides per "
        "district. Hamirpur had the highest density in the state, about **178 per 100 km²**. I then slid the study "
        "box north to keep almost all of Hamirpur's slides and pick up the dense part of south Kangra."
    )
    box = st.radio("Study box", ["Hamirpur district only", "Original rectangle", "Final box (shifted north)"],
                   index=2, horizontal=True)
    counts = {"Hamirpur district only": (326, 3, 1666), "Original rectangle": (639, 5, 2570),
              "Final box (shifted north)": (698, 39, 3694)}[box]
    c = st.columns(4)
    c[0].metric("2014 slides", counts[0])
    c[1].metric("2017 slides", counts[1])
    c[2].metric("2023 slides", f"{counts[2]:,}")
    c[3].metric("Total landslides", f"{sum(counts):,}", delta=f"{sum(counts) - 1995:+,} vs district" if sum(counts) != 1995 else None)
    st.image(str(RESULTS / "fig_study_area.png"),
             caption="Study area: 76.28-76.75°E, 31.55-32.03°N, UTM zone 43N. Districts, relief, rivers and mapped landslides.")
    takeaway("I chose the area by the data, not by the map: 4,431 real landslides instead of a handful.")
    notes(
        "Switch the radio between the three boxes: total goes 1,995 → 3,214 → 4,431.",
        "Say: for ML, the number of real examples matters more than anything else.",
        "Mention 2017 adds very few slides, so in practice the inventory is 2014 + 2023 (I say this again in limits).",
    )
    nav()

# =====================================================================
# 3. Database
# =====================================================================
elif ch == 3:
    st.header("3 · Building the database: what is different at a landslide?")
    st.markdown(
        "For every landslide I placed one point, and the same number of **non-landslide points** 0.5-3 km away "
        "(close enough to be on similar, well-mapped ground, far enough not to be on a slide). "
        "At each of the **8,696 points** I sampled 13 factors from Google Earth Engine: terrain from SRTM, drainage "
        "from MERIT Hydro, roads, soil clay, CHIRPS rainfall and three Landsat / Sentinel-2 indices. Each slide "
        "was sampled on imagery from *before* its season."
    )
    feat = st.selectbox("Pick a factor and compare landslide vs non-landslide points",
                        ["slope", "mean_annual_rainfall_mm", "elevation", "distance_to_road", "ndvi", "ndbi",
                         "distance_to_river", "twi", "soil_clay"], format_func=lambda f: NICE[f])
    d = points[[feat, "label"]].dropna().copy()
    d["Point"] = np.where(d.label == 1, "Landslide", "Non-landslide")
    lo, hi = d[feat].quantile([0.01, 0.99])
    d = d[(d[feat] >= lo) & (d[feat] <= hi)]
    dens = alt.Chart(d).transform_density(feat, groupby=["Point"], as_=[feat, "density"]).mark_area(opacity=0.55).encode(
        x=alt.X(f"{feat}:Q", title=f"{NICE[feat]} ({UNITS.get(feat, 'index')})"),
        y=alt.Y("density:Q", title="Share of points"),
        color=alt.Color("Point:N", scale=alt.Scale(domain=["Landslide", "Non-landslide"], range=["#d7191c", "#2c7bb6"]),
                        legend=alt.Legend(orient="top", title=None)),
    ).properties(height=320)
    st.altair_chart(dens, width="stretch")
    m1, m0 = d[d.label == 1][feat].median(), d[d.label == 0][feat].median()
    st.caption(f"Median at landslides: **{m1:,.2f}** · at non-landslide points: **{m0:,.2f}**. "
               "The more the two curves separate, the more useful the factor is to a model.")

    st.subheader("A data trap I caught: the CHIRPS rainfall jump")
    rain = change[["year", "rain_annual_mm", "rain_era5_annual_mm"]].rename(
        columns={"rain_annual_mm": "CHIRPS", "rain_era5_annual_mm": "ERA5-Land"}).melt("year", var_name="Dataset", value_name="mm")
    rc = alt.Chart(rain).mark_line(point=True).encode(
        x=alt.X("year:O", title=None), y=alt.Y("mm:Q", title="Mean annual rainfall (mm)"),
        color=alt.Color("Dataset:N", legend=alt.Legend(orient="top", title=None)), tooltip=["year", "Dataset", alt.Tooltip("mm:Q", format=",.0f")])
    rule = alt.Chart(pd.DataFrame({"year": [2009]})).mark_rule(strokeDash=[4, 4], color="grey").encode(x="year:O")
    st.altair_chart((rc + rule).properties(height=260), width="stretch")
    st.markdown(
        "CHIRPS shows about **466 mm/year before 2009 and 1,862 mm after**, a fourfold jump that is not real for "
        "these hills. ERA5-Land, a different dataset, shows only 1,350 → 1,604 mm. So I built the rainfall factor "
        "from **2009-2025 only**, which still covers all three landslide seasons."
    )
    takeaway("slope separates landslide from non-landslide ground most clearly, and checking the data saved me from a fake rainfall signal.")
    notes(
        "Open Slope first: landslides sit on clearly steeper ground. Then Rainfall and Distance to road.",
        "Explain the 0.5-3 km rule for non-landslide points (Bhuvan maps some areas better than others).",
        "Rainfall trap: point at the 2009 jump; ERA5 is the cross-check. Good point for the hydrology question.",
        "If asked about lithology: no digital geology layer in GEE, Bhukosh is toposheet-by-toposheet, so it's a stated limit (chapter 6 shows rock type at slides).",
    )
    nav()

# =====================================================================
# 4. Testing trap
# =====================================================================
elif ch == 4:
    st.header("4 · The testing trap: a random split flatters the model")
    st.markdown(
        "**Problem:** neighbouring points have almost the same slope, rain and land cover. If I split points "
        "randomly, the test points sit right next to training points, so a model can score well by memorising "
        "places. But the map is used on ground with no recorded landslides.\n\n"
        "**What I did:** tested every model three ways. Move the slider to see what happens to the ranking."
    )
    test = st.select_slider("How should we test the models?",
                            options=["A: random 70/30", "B: spatial 5 km blocks", "C: train 2014+2017 → test 2023"],
                            value="A: random 70/30")
    t = test[0]
    sel = models[models.Short == t].copy()
    sel["Rank"] = sel.ROC_AUC.rank(ascending=False).astype(int)
    left, right = st.columns([3, 2])
    with left:
        bars = alt.Chart(sel).mark_bar().encode(
            x=alt.X("Model:N", sort=MODEL_ORDER, title=None, axis=alt.Axis(labelAngle=0, labelFontSize=13)),
            y=alt.Y("ROC_AUC:Q", scale=alt.Scale(domain=[0.7, 1.0], clamp=True), title="ROC-AUC", axis=alt.Axis(format=".2f")),
            color=alt.condition(alt.datum.Model == "XGBoost", alt.value("#d7191c"), alt.value("#2c7bb6")),
            tooltip=["Model", alt.Tooltip("ROC_AUC:Q", format=".3f"), "Rank"])
        text = bars.mark_text(dy=-8).encode(text=alt.Text("ROC_AUC:Q", format=".3f"), color=alt.value("black"))
        st.altair_chart((bars + text).properties(height=340), width="stretch")
    with right:
        # little picture of the two splits, drawn on a sample of the real points
        s = points.sample(1500, random_state=1)[["lon", "lat"]].copy()
        if t == "A":
            s["Set"] = np.where(np.random.default_rng(0).random(len(s)) < 0.3, "Test", "Train")
            cap = "Random: test points (red) sit between training points."
        else:
            x_km, y_km = (s.lon - s.lon.min()) * 94.8, (s.lat - s.lat.min()) * 111
            bx, by = (x_km // 5).astype(int), (y_km // 5).astype(int)
            s["Set"] = np.where((bx * 3 + by * 2) % 5 == 0, "Test", "Train")  # one fold of a 5-fold block split
            cap = "Blocks: whole 5 km squares are held out, so the model is tested on new ground."
            if t == "C":
                cap = "Temporal: train on 2014 + 2017 slides, test on the 2023 slides (a future season)."
        st.altair_chart(alt.Chart(s).mark_circle(size=12).encode(
            x=alt.X("lon:Q", scale=alt.Scale(zero=False), axis=None), y=alt.Y("lat:Q", scale=alt.Scale(zero=False), axis=None),
            color=alt.Color("Set:N", scale=alt.Scale(domain=["Train", "Test"], range=["#bbbbbb", "#d7191c"]), legend=alt.Legend(orient="top", title=None)),
        ).properties(height=320), width="stretch")
        st.caption(cap)
    xa, xb = auc("XGBoost", "A"), auc("XGBoost", "B")
    if t == "A":
        st.info(f"On a random split every model looks excellent, and **XGBoost wins with {xa:.3f}**. This is the kind of number that looks impressive, and it is misleading.")
    elif t == "B":
        st.warning(f"Hold out whole 5 km blocks and **XGBoost falls to last ({xb:.3f}, a drop of {round(xa, 3) - round(xb, 3):.3f})**: it was partly memorising locations. ANN and SVM hold up best.")
    else:
        st.info("Predicting a future season is the hardest test. Every model drops, but all still rank 2023 slides well above chance (0.5).")
    takeaway("how you test matters more than which algorithm you pick, so I report the spatial result (Test B) as the honest number.")
    notes(
        "Start on A: everyone ~0.92-0.97, XGBoost best. Ask: would you trust this?",
        "Slide to B: XGBoost drops 0.116 to last. Explain spatial autocorrelation in one sentence.",
        "Slide to C: training on old seasons, predicting 2023: 0.77-0.82, harder but still well above chance.",
        "This is the key idea of the project; the intern's 99% on Kedarnath was this trap plus too few samples.",
    )
    nav()

# =====================================================================
# 5. Picking the model
# =====================================================================
elif ch == 5:
    st.header("5 · Picking the model: tune on the honest test")
    st.markdown(
        "I kept 15% of points aside and tuned each model with 20 random settings, scoring every setting with "
        "**spatial 5 km cross-validation**, not a random split. Then I picked the model that did best on new ground."
    )
    sp = tuning[tuning.Test == "Spatial 5 km CV"].pivot_table(index="Model", columns="Version", values="ROC_AUC").reindex(MODEL_ORDER).reset_index()
    base = alt.Chart(sp).encode(y=alt.Y("Model:N", sort=MODEL_ORDER, title=None))
    rule = base.mark_rule(color="#999").encode(x=alt.X("Default:Q", scale=alt.Scale(domain=[0.84, 0.92]), title="Spatial-CV ROC-AUC"), x2="Tuned:Q")
    p_def = base.mark_circle(size=140, color="#bbbbbb").encode(x="Default:Q", tooltip=[alt.Tooltip("Default:Q", format=".3f")])
    p_tun = base.mark_circle(size=180).encode(x="Tuned:Q", color=alt.condition(alt.datum.Model == "SVM", alt.value("#d7191c"), alt.value("#2c7bb6")),
                                               tooltip=[alt.Tooltip("Tuned:Q", format=".3f")])
    st.altair_chart((rule + p_def + p_tun).properties(height=280), width="stretch")
    st.caption("Grey = default settings, colour = tuned. Red = the model I chose.")
    c = st.columns(4)
    c[0].metric("Chosen model", "SVM (RBF)")
    c[1].metric("Spatial ROC-AUC", f"{svm_spatial.ROC_AUC:.3f}")
    c[2].metric("Overall accuracy", f"{svm_spatial.OA * 100:.1f}%")
    c[3].metric("Kappa", f"{svm_spatial.Kappa:.2f}")
    st.markdown(f"Settings found: C = {settings.loc['SVM', 'C']:.2f}, gamma = {settings.loc['SVM', 'gamma']:.3f}. "
                "A smooth decision boundary that carries over well to new areas. All tuned models are within about "
                "0.05 AUC of each other, so the choice of model matters less than the testing design.")
    takeaway("the tuned SVM is the best on unseen ground (AUC 0.905, Kappa 0.66, 'substantial agreement'), so it makes the final map.")
    notes(
        "Point at SVM moving right to 0.905; XGBoost improves but stays lowest.",
        "Kappa 0.66 = substantial agreement on the Landis and Koch scale.",
        "If asked why not an ensemble: tuned models are within 0.05 AUC, so gains would be small; possible future work.",
    )
    nav()

# =====================================================================
# 6. Drivers
# =====================================================================
elif ch == 6:
    st.header("6 · What drives landslides here?")
    st.markdown("I shuffled one factor at a time and measured how much the ROC-AUC dropped. A bigger drop means the model leans on that factor more.")
    imp = importance.copy()
    imp["Factor"] = imp.Factor.map(NICE)
    long = imp.melt("Factor", var_name="Model", value_name="AUC drop")
    order = imp.sort_values("Random Forest", ascending=False).Factor.tolist()
    st.altair_chart(alt.Chart(long).mark_bar().encode(
        y=alt.Y("Factor:N", sort=order, title=None), x=alt.X("AUC drop:Q", title="Drop in ROC-AUC when shuffled"),
        color=alt.Color("Model:N", legend=alt.Legend(orient="top", title=None)), yOffset="Model:N",
        tooltip=["Factor", "Model", alt.Tooltip("AUC drop:Q", format=".3f")]).properties(height=430), width="stretch")
    st.markdown(
        "**Slope** is the most consistent factor, as expected from slope-stability mechanics (shear stress grows "
        "on steeper ground). **Northness** comes next: north-facing slopes stay wetter. **Rainfall** ranks high, "
        "but the 5.5 km CHIRPS grid has few distinct values here, so it may partly act as a location label."
    )
    left, right = st.columns(2)
    with left:
        st.subheader("Rock type at the slides")
        l = lith.head(5).copy()
        l["Share"] = l.slides / lith.slides.sum() * 100
        l["Lithology"] = l.lithology.str.title().str.slice(0, 45)
        st.altair_chart(alt.Chart(l).mark_bar(color="#8c6d31").encode(
            y=alt.Y("Lithology:N", sort="-x", title=None), x=alt.X("Share:Q", title="% of mapped landslides"),
            tooltip=["Lithology", alt.Tooltip("Share:Q", format=".1f")]).properties(height=240), width="stretch")
        st.caption("Lithology was not available as a digital layer, so it is not a model factor. The Bhuvan records show "
                   "~77% of slides in loosely cemented sandstone-conglomerate units.")
    with right:
        st.subheader("Was vegetation loss to blame?")
        v = veg.copy()
        v["Class"] = ["Q1 least greening", "Q2", "Q3", "Q4", "Q5 most greening"]
        st.altair_chart(alt.Chart(v).mark_bar(color="#1a9641").encode(
            x=alt.X("Class:N", sort=None, title="NDVI change 2005-07 → 2019-21"), y=alt.Y("Frequency_ratio:Q", title="Frequency ratio, 2023 slides"),
            tooltip=["Class", alt.Tooltip("Frequency_ratio:Q", format=".2f")]).properties(height=240), width="stretch")
        st.caption("The area got greener (MODIS NDVI ~0.55 → 0.65), and 2023 slides were not concentrated where vegetation was lost. "
                   "At 250 m, the deforestation idea is not supported.")
    takeaway("slope controls where slides happen, soft Shivalik rocks make it worse, and vegetation loss was not the cause at this scale.")
    notes(
        "Slope first for both models; rainfall high for XGBoost with the CHIRPS caveat.",
        "Lithology: be upfront it's not in the model, show the 77% in conglomerate/sandstone (Guha Roy will ask).",
        "Land-cover change was my hypothesis; I tested it and the data said no. A negative result is still a result.",
    )
    nav()

# =====================================================================
# 7. Map
# =====================================================================
elif ch == 7:
    st.header("7 · The susceptibility map")
    image, b = load_overlay()
    show_slides = st.checkbox("Show mapped landslide points", value=False)
    layers = [pdk.Layer("GeoJsonLayer", load_districts(), stroked=True, filled=False, get_line_color=[60, 60, 60], line_width_min_pixels=1),
              pdk.Layer("BitmapLayer", image=pdk.types.String(image),
                        bounds=[[b["west"], b["south"]], [b["west"], b["north"]], [b["east"], b["north"]], [b["east"], b["south"]]])]
    if show_slides:
        layers.append(pdk.Layer("ScatterplotLayer", points[points.label == 1][["lon", "lat"]], get_position=["lon", "lat"],
                                get_fill_color=[90, 40, 140, 160], get_radius=60))
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(latitude=31.79, longitude=76.52, zoom=9.3),
                             map_style="light"), height=520)
    st.markdown(" ".join(f"<span style='background:{c};padding:2px 8px;border:1px solid #999'>{z}</span>"
                         for z, c in zip(ZONE_ORDER, ZONE_COLOURS)), unsafe_allow_html=True)

    st.subheader("Does the map put the slides in the right places?")
    cut = st.select_slider("Flag every zone from ... upwards", options=ZONE_ORDER[1:], value="High")
    flagged = zones.iloc[ZONE_ORDER.index(cut):]
    c = st.columns(3)
    c[0].metric("Area flagged", f"{flagged['Area_%'].sum():.1f}%")
    c[1].metric("Mapped landslides inside", f"{flagged['Landslides_%'].sum():.1f}%")
    c[2].metric("Landslides per unit area vs average", f"{flagged['Landslides_%'].sum() / flagged['Area_%'].sum():.1f}×")
    st.altair_chart(alt.Chart(zones).mark_bar().encode(
        x=alt.X("Zone:N", sort=ZONE_ORDER, title=None), y=alt.Y("Frequency_ratio:Q", title="Frequency ratio"),
        color=alt.Color("Zone:N", sort=ZONE_ORDER, legend=None, scale=alt.Scale(domain=ZONE_ORDER, range=ZONE_COLOURS)),
        tooltip=["Zone", alt.Tooltip("Area_%:Q", format=".1f"), alt.Tooltip("Landslides_%:Q", format=".1f"),
                 alt.Tooltip("Frequency_ratio:Q", format=".2f")]).properties(height=240), width="stretch")
    st.caption("Frequency ratio = share of slides in a zone ÷ share of area. It rises from 0.06 to 6.45. These are training "
               "slides, so this is a consistency check; the independent test is the spatial AUC from chapter 5.")
    takeaway(f"High + Very High cover {hv['Area_%'].sum():.1f}% of the area but hold {hv['Landslides_%'].sum():.1f}% of the mapped landslides.")
    notes(
        "Zoom in on a red patch: point out it follows steep valley sides and road cuts, not the flat terraces.",
        "Slider: at 'High' → 19.9% of area, 83.3% of slides; at 'Very High' → 10.4% of area, 67.3% of slides.",
        "Be honest: 2024 imagery shows fresh 2023 scars as bare ground, so it can raise local scores.",
    )
    nav()

# =====================================================================
# 8. Exposure
# =====================================================================
elif ch == 8:
    st.header("8 · Who is exposed, and what should be done?")
    left, right = st.columns(2)
    with left:
        what = st.radio("Show share of", ["Road_%", "Population_%", "Built_%", "Area_%"], horizontal=True,
                        format_func=lambda s: {"Road_%": "major roads", "Population_%": "people", "Built_%": "built-up land", "Area_%": "area"}[s])
        st.altair_chart(alt.Chart(exposure).mark_bar().encode(
            x=alt.X("Zone:N", sort=ZONE_ORDER, title=None), y=alt.Y(f"{what}:Q", title="% of study-area total"),
            color=alt.Color("Zone:N", sort=ZONE_ORDER, legend=None, scale=alt.Scale(domain=ZONE_ORDER, range=ZONE_COLOURS)),
            tooltip=["Zone", "Road_km", "Population", "Built_up_km2"]).properties(height=280), width="stretch")
        st.caption(f"High + Very High: {exp_hv.Road_km.sum():.1f} km of major roads, ~{exp_hv.Population.sum():,.0f} people, "
                   f"but only {exp_hv['Built_%'].sum():.1f}% of built-up land. Towns sit on the flatter valley floors and terraces; roads must cross the hills.")
    with right:
        dd = districts.sort_values("High_VeryHigh_%", ascending=False)
        st.altair_chart(alt.Chart(dd).mark_bar(color="#fdae61").encode(
            y=alt.Y("District:N", sort="-x", title=None), x=alt.X("High_VeryHigh_%:Q", title="% of district (inside box) in High + Very High"),
            tooltip=["District", alt.Tooltip("High_VeryHigh_%:Q", format=".1f"), alt.Tooltip("People_in_High_VeryHigh:Q", format=",.0f")]).properties(height=280),
            width="stretch")
        st.caption("Kangra's part of the box is the most exposed (28%), then Mandi and Hamirpur.")

    st.subheader("Priority road stretches")
    share = st.slider("Only stretches with at least this share in the Very High zone (%)", 50, 90, 50, 5)
    r = roads[roads["share_very_high_%"] >= share]
    c = st.columns(3)
    c[0].metric("Stretches", len(r))
    c[1].metric("Total length", f"{r.length_m.sum() / 1000:.1f} km")
    c[2].metric("Longest", f"{r.length_m.max():.0f} m" if len(r) else "-")
    st.dataframe(r.rename(columns={"road_type": "Road type", "length_m": "Length (m)", "share_very_high_%": "% in Very High",
                                   "mid_lat": "Lat", "mid_lon": "Lon"}).round(4), hide_index=True, width="stretch", height=210)

    st.subheader("The risk framework")
    st.table(pd.DataFrame({
        "Zone": ["Very High", "High", "Moderate", "Low / Very Low"],
        "Planning and land use": ["Avoid new construction and new road alignments", "Develop only after a site geotechnical investigation",
                                  "Normal development with slope checks for cuts and fills", "Preferred for new settlements and infrastructure"],
        "Slopes and roads": ["Stabilise road cuts first, drainage, pre/post-monsoon inspection, monitoring", "Engineered cut slopes, drainage, bio-engineering; maintenance priority",
                             "Maintain drainage; avoid over-steep cuts", "Standard good practice"],
    }).set_index("Zone"))
    takeaway("the risk is mainly to roads: 40 stretches (17.5 km) through High / Very High ground are where slope protection money should go first.")
    notes(
        "Roads vs built-up contrast: 10.5% of roads but only 1% of built-up land is in High + VH.",
        "Move the slider up to 90%: the most critical stretches shrink to a short list a PWD engineer could act on.",
        "The framework turns zones into actions: this is Objective 5.",
    )
    nav()

# =====================================================================
# 9. What if
# =====================================================================
elif ch == 9:
    st.header("9 · What if? Run the final model yourself")
    st.markdown("Pick a real point from the dataset, then change its factors and watch the predicted probability move.")
    bundle = load_model()
    pts = add_aspect_terms(points)
    kind = st.radio("Start from", ["A landslide point", "A non-landslide point"], horizontal=True)
    pool = pts[pts.label == (1 if kind == "A landslide point" else 0)].reset_index(drop=True)
    idx = st.slider("Point number", 0, len(pool) - 1, 0)
    row = pool.loc[idx]
    sliders = {
        "slope": ("Slope (°)", 0.0, 67.0), "mean_annual_rainfall_mm": ("Mean annual rainfall (mm)", 1550.0, 2500.0),
        "elevation": ("Elevation (m)", 450.0, 1560.0), "distance_to_road": ("Distance to road (m)", 0.0, 5000.0),
        "ndvi": ("NDVI (vegetation)", -0.17, 0.82), "aspect": ("Aspect (° from north)", 0.0, 359.0),
    }
    edited = row.copy()
    cols = st.columns(3)
    for i, (f, (label, lo, hi)) in enumerate(sliders.items()):
        edited[f] = cols[i % 3].slider(label, lo, hi, float(np.clip(row[f], lo, hi)), key=f"{f}_{kind}_{idx}")
    raw = [f for f in bundle["features"] if f not in ("northness", "eastness")] + ["aspect"]

    def predict(df):
        X = add_aspect_terms(df[raw].astype(float))[bundle["features"]]
        return bundle["model"].predict_proba(X.to_numpy())[:, 1]

    prob = float(predict(pd.DataFrame([edited]))[0])
    zone = ZONE_ORDER[min(int(prob // 0.2), 4)]
    a, b_, c_ = st.columns(3)
    a.metric("Predicted landslide probability", f"{prob:.2f}")
    b_.metric("Zone", zone)
    c_.metric("Actually a landslide?", "Yes" if row.label == 1 else "No", f"{int(row.season)} inventory" if row.label == 1 else None)

    sweep = pd.DataFrame([edited] * 35)
    sweep["slope"] = np.linspace(0, 67, 35)
    sweep["Probability"] = predict(sweep)
    line = alt.Chart(sweep).mark_line(color="#d7191c").encode(
        x=alt.X("slope:Q", title="Slope (°), all other factors held at this point's values"),
        y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1])))
    band = alt.Chart(pd.DataFrame({"y": [0.2, 0.4, 0.6, 0.8]})).mark_rule(strokeDash=[3, 3], color="#bbb").encode(y="y:Q")
    dot = alt.Chart(pd.DataFrame({"slope": [edited["slope"]], "Probability": [prob]})).mark_circle(size=160, color="black").encode(x="slope:Q", y="Probability:Q")
    st.altair_chart((band + line + dot).properties(height=260, title="How probability changes with slope at this spot"), width="stretch")
    st.caption(f"Location {row.lat:.4f}°N, {row.lon:.4f}°E. Probabilities come from the calibrated SVM (CalibratedClassifierCV). Dashed lines are the zone boundaries.")
    takeaway("the model behaves the way slope mechanics says it should: steeper ground, higher probability, all else equal.")
    notes(
        "Start from a landslide point (high probability), then drag slope down: watch it fall through the zones.",
        "Switch to a non-landslide point and increase slope: it climbs.",
        "Offer the panel: 'Give me a slope and rainfall and I'll show you what the model says.'",
    )
    nav()

# =====================================================================
# 10. Conclusions
# =====================================================================
else:
    st.header("10 · What I achieved, and what is still open")
    st.markdown("**Objectives**")
    st.table(pd.DataFrame({
        "Objective": ["1. GIS database", "2. Controlling factors", "3. Susceptibility map", "4. ML models", "5. Risk framework"],
        "What I delivered": [
            "13 factors for 3 epochs (~2006, ~2016, ~2024) in Earth Engine + 4,431 Bhuvan landslides",
            "Slope, northness and rainfall lead; ~77% of slides in sandstone-conglomerate; no link to vegetation loss",
            "30 m map, five zones; High + Very High = 19.9% of area with 83.3% of slides",
            "kNN, SVM, ANN, RF, XGBoost tested 3 ways; tuned SVM best on new ground (AUC 0.905)",
            "Roads, people, built-up land per zone; 40 priority road stretches; zone-wise actions",
        ],
        "Status": ["✅", "✅ (lithology only descriptive)", "✅", "✅", "✅"],
    }).set_index("Objective"))
    left, right = st.columns(2)
    with left:
        st.markdown(
            "**Limitations (stated honestly)**\n"
            "- Inventory is effectively 2014 + 2023 (~83% from 2023), a 9-year span.\n"
            "- Lithology not a model factor (no digital layer); soil and rainfall are coarse.\n"
            "- Geotechnical strength parameters (c, φ) and their change over time are not modelled.\n"
            "- 2024 imagery shows fresh 2023 scars; no field check yet."
        )
    with right:
        st.markdown(
            "**Next steps**\n"
            "- Add GSI 1:50,000 geology and IMD gridded rainfall.\n"
            "- More seasons of inventory and a field check of the map.\n"
            "- Link the zones to slope-stability analysis with measured c and φ.\n"
            "- InSAR to catch slow-moving slopes before they fail."
        )
    st.info("**One line to remember:** test the model on ground it has never seen. On a random split XGBoost looked best; "
            "on new ground the tuned SVM was, and its map puts 83% of mapped slides into 20% of the area.")
    notes(
        "Tick through the 5 objectives in 30 seconds.",
        "Limitations before they ask, especially c/φ (Guha Roy) and rainfall data (Swami).",
        "End on the one line, then: 'Thank you, happy to take questions. I can show any chapter again.'",
    )
    nav()
