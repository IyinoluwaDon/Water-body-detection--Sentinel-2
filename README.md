# Sentinel-2 Water Mapping with a Spatially Evaluated Decision Tree

A geospatial machine-learning workflow for seasonal surface-water mapping in the Lagos lagoon study area. It builds 10 m seasonal composites from real Sentinel-2 imagery, derives spectral/index features, trains an interpretable decision tree from historical water labels, and produces dry/wet maps and a change summary. The Streamlit app presents generated results; the core work is the Earth Engine export, model training, spatial evaluation, and raster pipeline in `src/waterdet/`.

## Data, features, and model

Google Earth Engine supplies Sentinel-2 L2A surface reflectance (`COPERNICUS/S2_SR_HARMONIZED`) with Cloud Score+ masking. The model uses B2, B3, B4, B8, B11, NDWI, and MNDWI. Training labels are derived from JRC Global Surface Water v1.4; an optional comparison mode also requires agreement with Google Dynamic World. Samples are assigned to 2 km spatial blocks, and the tuned scikit-learn `DecisionTreeClassifier` is selected with five-fold GroupKFold and a one-standard-error simplicity rule.

The recorded Lagos run uses the dry-season imagery to train and applies the model to dry and wet composites. Its primary GSW-label run reports a depth-2 tree and **98.52% agreement with JRC-derived labels on held-out spatial blocks** (95% block-bootstrap interval 96.9-99.5%). The strongest same-sample threshold baseline, MNDWI Otsu, reports 98.79% agreement. The run maps 230.10 km2 of water in the dry composite and 236.88 km2 in the wet composite, with 10.56 km2 classified as gained and 3.80 km2 as lost on comparable coverage. These are comparisons against the historical label source, not a manually labelled ground survey.

`docs/METHODOLOGY.md` details the satellite processing, label construction, split design, baselines, and metric definitions. `outputs/` and `results/` preserve run summaries, tree rules, maps, and reproducibility records. The Lagos results use real exported Sentinel-2 imagery and JRC labels; `src/waterdet/synthetic.py` creates a separate known-truth synthetic world for pipeline tests and does not supply the reported Lagos data.

## Run

```sh
python -m pip install -e ".[gee,dev,app]"
python -m waterdet.demo --out-dir outputs/demo
python -m streamlit run app/streamlit_app.py
```

For a real Earth Engine export and model run, follow `docs/HOW_TO_RUN.md` and `docs/METHODOLOGY.md`. The project configuration and test suite are in `config.yaml` and `tests/`.
