# Landslide Susceptibility Mapping with GIS and Machine Learning

**B.Tech Project, Department of Civil and Infrastructure Engineering, IIT Jodhpur (2026)**
Subham Agrawal (B23CI1038) · Supervisor: Prof. Susmita Panda · Co-supervisor: Prof. Nallala Sarath Chandra Reddy

Where in the Hamirpur and south Kangra hills (Himachal Pradesh) are landslides most likely, and which roads and people sit on that ground?
This project answers that with satellite data in Google Earth Engine and five machine-learning classifiers, tested the way a map like this is actually used: on areas the model has never seen.

![Final susceptibility map](study_area/Results/fig_susceptibility_final.png)

## Headline results

| | |
|---|---|
| Study area | 76.28-76.75°E, 31.55-32.03°N, about 2,360 km² (Hamirpur + parts of Kangra, Mandi, Una, Bilaspur) |
| Landslide inventory | 4,431 mapped landslides (NRSC Bhuvan, 2014 / 2017 / 2023 monsoons) |
| Training data | 8,696 points (landslide and non-landslide), 14 conditioning factors |
| Final model | Tuned SVM (RBF, C = 1.31, gamma = 0.027) |
| Spatial-CV performance | ROC-AUC **0.905**, overall accuracy 83.0%, Cohen's kappa 0.66 |
| Map | 2.62 million 30 m pixels; 19.9% of the area is High or Very High |
| Frequency ratio | 0.06 (Very Low) to 6.45 (Very High); 83.3% of mapped slides fall in High + Very High |
| Exposure | High + Very High zones hold 71.9 km of major roads and about 57,000 people; 40 road stretches (17.5 km) flagged for slope protection |

## The key finding: random splits overstate accuracy

Nearby points have almost identical slope, rainfall and land cover, so a random train/test split lets a model "memorise" places. Every model was therefore tested three ways:

| Model | A: random 70/30 | B: spatial 5 km blocks | C: train 2014+2017, test 2023 |
|---|---|---|---|
| kNN | 0.921 | 0.876 | 0.765 |
| SVM | 0.933 | 0.886 | 0.782 |
| ANN | 0.924 | **0.901** | 0.767 |
| Random Forest | 0.932 | 0.875 | 0.813 |
| XGBoost | **0.967** | 0.851 | **0.824** |

(ROC-AUC, default settings; `study_area/Results/model_comparison.csv`)

XGBoost wins the random split but drops to last when whole 5 km blocks are held out: it loses the most (0.116 AUC), so it was partly memorising locations.
Test B is the one that matters, because the map is used for ground with no recorded landslides. After tuning on spatial CV, the SVM has the best spatial AUC (0.905) and is the final model.
All tuned models are within 0.05 AUC of each other, so the sampling and testing design matters more than the choice of algorithm.

## Pipeline

```
Bhuvan landslide polygons ─┐
                           ├─> landslide + non-landslide points ─> factor values at each point
Earth Engine factor stack ─┘        (pre-event imagery)                      │
 (Landsat / Sentinel-2, SRTM,                                                v
  MERIT Hydro, GRIP4, OpenLandMap,           5 models x 3 tests (random, spatial block, temporal)
  CHIRPS)                                                                    │
                                                                             v
                                    tune on spatial CV ─> best model (SVM) ─> 30 m susceptibility map
                                                                             │
                                                                             v
                                    5 zones + frequency ratio ─> roads / built-up / population exposure
```

Everything runs in one notebook, [`study_area/Hamirpur.ipynb`](study_area/Hamirpur.ipynb). Each code cell is followed by a short "Result" note explaining its output.

| Cells | What they do |
|---|---|
| 1-2 | Earth Engine login; study area box; UTM 43N projection (EPSG:32643) |
| 3-5 | Cloud/shadow masks; median composites for ~2006 (Landsat 5/7), ~2016 and ~2024 (Sentinel-2); NDVI, NDWI, NDBI |
| 6-8 | Terrain (SRTM 30 m: elevation, slope, aspect, plan curvature), hydrology (MERIT Hydro: TWI, drainage density, distance to river), distance to road (GRIP4), soil clay (OpenLandMap), mean annual rainfall 2009-2025 (CHIRPS); 13-band stack per period |
| 9 | Interactive map of all layers (geemap) |
| 10 | Landslide points from the Bhuvan inventory, de-duplicated per 30 m pixel (4,353); equal number of non-landslide points 0.5-3 km from the nearest slide |
| 11 | Factor values sampled at every point from imagery taken **before** each landslide season (2014 uses ~2006, 2017 and 2023 use ~2016) |
| 12 | kNN, SVM, ANN, Random Forest, XGBoost under tests A, B, C; ROC curves, confusion matrices, permutation importance |
| 13 | Random search (20 settings per model) scored by 5-fold spatial block CV, 15% test set held out; saves `best_model.joblib` |
| 14 | Best model applied to every 30 m pixel (2024 imagery); five zones (0.2 probability steps); frequency ratio |
| 15 | 600 dpi report maps |
| 16 | Context: MODIS NDVI trend, CHIRPS vs ERA5-Land rainfall, GHSL population, USGS earthquakes within 150 km |
| 17 | Risk framework: roads, built-up land and population in each zone, by district; priority road stretches |

`study_area/Bhuvan_inventory/bhuvan_pull_hp_inventory.py` is the script (run as a cell before Cell 10) that pulls the Bhuvan seasonal landslide inventory. Bhuvan has no download button, but its map server answers WMS `GetFeatureInfo`; asking for the whole state as one 1x1 pixel returns every polygon with its attributes.

## Design choices worth knowing

- **Aspect is circular** (0° and 359° are both north), so it enters the models as northness (cos) and eastness (sin): 14 inputs from 13 layers.
- **Pre-event imagery** for each point, so the model cannot just see fresh landslide scars.
- **Same-pixel duplicates merged**, so one location cannot sit in both train and test.
- **Non-landslide points 0.5-3 km from a slide**: off the slides, but inside the area Bhuvan actually mapped.
- **Rainfall from 2009 onward only**: CHIRPS gives about 466 mm/yr here for 2001-2008 against 1,862 mm/yr after, a jump ERA5-Land does not show (1,350 vs 1,604 mm). It is a data problem, not a climate change.
- **SVM probabilities** come from `CalibratedClassifierCV`, needed for ROC-AUC and for the probability map.

## Which factors matter

Permutation importance (drop in ROC-AUC when one factor is shuffled, on the test-A split): slope ranks first for Random Forest and second for XGBoost; rainfall first for XGBoost; northness second for Random Forest. TWI, drainage density and soil clay matter least, probably because their source data are coarse (90-250 m).
Caveat: the 5.5 km CHIRPS grid gives only a few distinct rainfall values over the area, so rainfall may partly act as a location label.

![Factor importance](study_area/Results/factor_importance.png)

## Limitations

- The frequency-ratio check uses the same landslides the model was trained on, so it is a consistency check, not an independent test.
- No lithology layer: none is available in Earth Engine, although 76.6% of the slides are in coarse sandstone and boulder conglomerate units (Bhuvan attributes).
- 2023 dominates the inventory (about 83% of slides), and the ~2016 composite is effectively 2017 imagery because Sentinel-2 surface reflectance starts then over India.
- Geotechnical parameters (c, φ) are not yet included.

## Repository layout

```
study_area/
  Hamirpur.ipynb                 full analysis, Cells 1-17, with outputs
  Bhuvan_inventory/
    bhuvan_pull_hp_inventory.py  inventory download (Bhuvan WMS GetFeatureInfo)
    bhuvan_HP_landslides_2014_2017_2023.csv   all HP landslide centroids
    inventory_samples_AOI.csv    landslide + non-landslide points in the study area
    ml_dataset_AOI.csv           the ML table: 8,696 points x factors + label
  Results/                       tables (.csv), figures (.png), best_model.joblib, susceptibility_zones.tif
docs/                            project report and presentation
```

Not included (size): the raw Bhuvan GeoJSON pulls (~20 MB) and the continuous probability raster (`susceptibility_probability.tif`, ~10 MB). Both are regenerated by the notebook.

## Running it

```bash
pip install -r requirements.txt
cd study_area
jupyter notebook Hamirpur.ipynb
```

Cells 1-11 and 14-17 need a Google Earth Engine account; change the project id in Cell 1 to your own. Cells 12-13 can run from `Bhuvan_inventory/ml_dataset_AOI.csv` alone (they reload it if the kernel was restarted), but they use `FEATURE_BANDS` from Cell 8.

## Data sources

NRSC/ISRO Bhuvan (landslide inventory) · USGS Landsat 5/7 and ESA Sentinel-2 · NASA SRTM · MERIT Hydro (Yamazaki et al., 2019) · GRIP4 roads (Meijer et al., 2018) · OpenLandMap soil clay · CHIRPS (Funk et al., 2015) · ERA5-Land · MODIS NDVI · GHSL population · ESA WorldCover · USGS ANSS earthquake catalogue.
