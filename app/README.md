# Web app

A small read-only website for this project, built with [Streamlit](https://streamlit.io).
It does not re-run the analysis: it reads the tables, figures and model the analysis already saved, and draws them as interactive charts.

## What `streamlit_app.py` does, in plain English

1. **Loads saved results.** Each `load_...` function reads one saved file. `@st.cache_data` means the file is read once when the server starts, not on every click.
2. **Header and metric cards.** The headline numbers are read from the result files, so they always match the report.
3. **Tabs.** Each `with tab_...:` block is one tab. Charts use Altair (hover for exact values); tables use `st.dataframe` (sortable).
4. **Widgets.** Radio buttons, checkboxes, sliders and dropdowns only change what is *shown*. Every time you click, Streamlit re-runs the script top to bottom with the new widget values; that is the whole programming model.
5. **Map tab.** `pydeck` draws a web map with three layers: district outlines (`hp_districts.geojson`), the susceptibility zones as a coloured image, and the 40 priority road stretches as dots. The zone image comes from `make_overlay.py`, which reprojects `susceptibility_zones.tif` from UTM (metres) to longitude/latitude once, because web maps work in degrees.
6. **Try the model tab.** Loads `best_model.joblib` (the tuned SVM pipeline: StandardScaler, then SVC with calibrated probabilities), takes one of the 8,696 training points, lets you move six of its factors, converts aspect to northness/eastness exactly as the notebook does, and shows the predicted probability and zone (0.2 steps).

## Run it on your laptop

```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

`app/requirements.txt` is separate from the notebook's `requirements.txt` so the website does not install Earth Engine or rasterio. scikit-learn is pinned to the version the saved model was made with.
