# Cell 10a - Pull Bhuvan Seasonal Landslide Inventory (HP) for 2014 / 2017 / 2023
# Paste this into a new cell just before Cell 10 and run it.
#
# How it works: Bhuvan's GeoServer has WFS switched off, but WMS GetFeatureInfo is on.
# If the whole state is squeezed into a single 1x1 pixel, one "click" on that pixel
# returns every polygon in the layer, attributes and geometry included.
# This was tested on HP_SLIM_2017 and it returned 172 polygons from all over HP.

import json
import urllib.request
import urllib.parse
import pandas as pd

WMS_URL = "https://bhuvan-vec2.nrsc.gov.in/bhuvan/wms"
HP_LAYERS = {
    2014: "disaster:HP_SLIM_2014_GCS",
    2017: "disaster:HP_SLIM_2017",
    2023: "disaster:LS_HIMACHALPRADESH_2023",
}
HP_BBOX = (75.5, 30.3, 79.1, 33.3)   # whole Himachal, lon/lat
MAX_FEATURES = 20000

# Study area box, same as STUDY_AREA_COORDS in Cell 1 (minlon, minlat, maxlon, maxlat)
AOI = STUDY_AREA_COORDS if 'STUDY_AREA_COORDS' in globals() else (76.28, 31.40, 76.75, 31.88)


def get_layer(layer_name):
    params = {
        "SERVICE": "WMS", "VERSION": "1.1.1", "REQUEST": "GetFeatureInfo",
        "LAYERS": layer_name, "QUERY_LAYERS": layer_name, "STYLES": "",
        "SRS": "EPSG:4326", "BBOX": ",".join(str(v) for v in HP_BBOX),
        "WIDTH": 1, "HEIGHT": 1, "X": 0, "Y": 0,
        "INFO_FORMAT": "application/json", "FEATURE_COUNT": MAX_FEATURES,
    }
    url = WMS_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def prop(props, *keys):
    # attribute names are not the same in every season's layer, so check a few spellings
    lower = {k.lower(): v for k, v in props.items()}
    for k in keys:
        if k.lower() in lower:
            return lower[k.lower()]
    return None


def centre_of(geom):
    # polygons here are small boxes, so the mean of the outer ring vertices is good enough
    pts = []
    if geom["type"] == "Point":
        return geom["coordinates"][0], geom["coordinates"][1]
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    for poly in polys:
        pts.extend(poly[0])
    lon = sum(p[0] for p in pts) / len(pts)
    lat = sum(p[1] for p in pts) / len(pts)
    return lon, lat


rows = []
all_features = []
for year, layer in HP_LAYERS.items():
    try:
        fc = get_layer(layer)
    except Exception as e:
        print(f"{year}: request failed -> {e}")
        continue
    feats = fc.get("features", [])
    print(f"{year} ({layer}): {len(feats)} polygons returned")
    if len(feats) >= MAX_FEATURES:
        print("   WARNING: hit FEATURE_COUNT cap, layer may be cut off - raise MAX_FEATURES")
    for f in feats:
        p = f.get("properties", {})
        lon, lat = centre_of(f["geometry"])
        rows.append({
            "name": prop(p, "SlideNo", "slide_no", "id") or f.get("id"),
            "lon": round(lon, 6),
            "lat": round(lat, 6),
            "source": f"Bhuvan_SLIM_{year}",
            "label": 1,
            "year": year,
            "district": str(prop(p, "District", "dist_name", "district_n") or "").upper(),
            "area_sqm": prop(p, "Area_Sqm", "area", "shape_area"),
        })
        f["properties"]["season"] = year
        all_features.append(f)

df_bhuvan = pd.DataFrame(rows)

# keep the raw polygons too - useful later for sampling several pixels per slide
with open("bhuvan_HP_all_seasons.geojson", "w") as fh:
    json.dump({"type": "FeatureCollection", "features": all_features}, fh)
df_bhuvan.to_csv("bhuvan_HP_all_seasons.csv", index=False)

in_aoi = df_bhuvan[
    (df_bhuvan.lon >= AOI[0]) & (df_bhuvan.lon <= AOI[2]) &
    (df_bhuvan.lat >= AOI[1]) & (df_bhuvan.lat <= AOI[3])
]
in_aoi.to_csv("hamirpur_landslide_points_bhuvan.csv", index=False)

print("\nAll HP, per season:")
print(df_bhuvan.groupby("year").size().to_string())
print("\nDistrict = HAMIRPUR, per season:")
print(df_bhuvan[df_bhuvan.district == "HAMIRPUR"].groupby("year").size().to_string())
print(f"\nInside the Hamirpur study box: {len(in_aoi)} polygons")
print(in_aoi.groupby("year").size().to_string())
print("\nTop 10 districts overall:")
print(df_bhuvan.district.value_counts().head(10).to_string())

# Cell 10 can now use these as real positives:
# curated_inventory = in_aoi[['name','lon','lat','source','label']].to_dict('records')
