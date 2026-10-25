# Technical notes: water-body detection

This is the implementation record for the geospatial and machine-learning terms used in the project. The public-facing scope and main findings remain in the [README](../README.md); reproducible steps are in [HOW_TO_RUN.md](HOW_TO_RUN.md) and methodological choices are in [METHODOLOGY.md](METHODOLOGY.md).

## Processing graph

The run is a raster-first workflow: Earth Engine exports seasonal Sentinel-2 surface-reflectance composites and quality masks; local code aligns the bands, derives spectral features, builds label and sample tables, tunes/evaluates a classifier, writes georeferenced maps, and packages lightweight assets for Streamlit. `config.yaml` is the run configuration, `src/waterdet/pipeline.py` orchestrates the local stages, `src/waterdet/gee_export.py` constructs the remote export graph, and `scripts/sync_notebook.py` embeds maintained source into the Colab notebook.

## Imagery and raster terminology

- **Sentinel-2 L2A surface reflectance:** atmospherically corrected optical reflectance. The project uses the harmonized Earth Engine collection `COPERNICUS/S2_SR_HARMONIZED`.
- **Cloud Score+:** a companion quality layer used to mask clouds and cloud shadows before compositing. The selected observations are reduced to a seasonal median, band by band.
- **Composite:** one representative multiband raster for a date window, assembled from multiple clear observations. It is not a single-day image.
- **Raster stack:** co-registered bands with the same dimensions, coordinate reference system (CRS), affine transform, and pixel grid. Alignment is required so a pixel's features and label describe the same location.
- **NoData / valid-data mask:** pixels that do not meet image-quality or denominator requirements are excluded, rather than silently assigned to land.
- **Affine transform:** maps raster row/column coordinates to map coordinates. Pixel area is computed from the absolute determinant of its 2×2 linear part; the pipeline requires a projected metric grid before reporting square kilometres.
- **20 m band on a 10 m grid:** Sentinel-2 B11 is natively coarser than the visible/NIR bands. It is aligned to the analysis grid for feature calculation; the resampling does not create new 10 m detail in B11.

## Spectral predictors

The feature vector contains reflectance bands B2 (blue), B3 (green), B4 (red), B8 (near infrared), and B11 (short-wave infrared), plus:

\[
NDWI = \frac{B3-B8}{B3+B8}, \qquad
MNDWI = \frac{B3-B11}{B3+B11}.
\]

These normalized-difference indices emphasize different water/land contrasts. A zero or invalid denominator produces an invalid pixel that is removed consistently from training and inference. Index values are dimensionless and lie approximately in [-1, 1] for valid nonnegative reflectance inputs.

## Reference labels and sampling

- **JRC Global Surface Water (GSW) v1.4:** the historical Landsat-derived water-occurrence reference. The project turns explicit water occurrence into water/land training classes and leaves missing/unobserved pixels unlabelled. Missing is not treated as land.
- **Dynamic World agreement mode:** an alternate label construction requires agreement between GSW and Dynamic World. This narrows the usable label set, so metrics from the two label modes have different denominators and should not be compared as if they were the same sample.
- **Confident core / erosion:** a morphology operation removes uncertain edge pixels from the training target. The shoreline is evaluated separately where the configured reference workflow supports it.
- **Spatial block:** a fixed geographic grid cell used as a group during sampling and validation. Keeping a whole block in one split reduces leakage from nearby, spatially autocorrelated pixels.
- **Balanced point sample:** the training/evaluation table samples class strata to make water and land adequately represented. Consequently, precision/F1 on this sample are not direct estimates at the natural mapped class prevalence.
- **Area-weighted reference sample:** the optional manual-check workflow weights each labelled stratum by its map-area share when estimating accuracy. This differs from simply averaging the sampled points.

The standard recorded run draws 10,000 candidate points, partitions by spatial blocks, uses grouped cross-validation for tree tuning, and reserves whole held-out blocks for the final comparison. `src/waterdet/sampling.py`, `labels.py`, and `reference.py` implement these steps.

## Classifier and validation

The fitted estimator is scikit-learn's `DecisionTreeClassifier`, with `class_weight="balanced"`. `src/waterdet/model.py` searches configured `max_depth` and `min_samples_leaf` settings using `GroupKFold` on spatial block IDs. The **one-standard-error rule** selects the simplest tree whose mean cross-validation score is within one estimated standard error of the best candidate; simplicity is ordered by smaller maximum depth, then larger leaf size.

`src/waterdet/evaluate.py` reports the confusion matrix with rows as reference classes and columns as predicted classes. For water, precision is user's accuracy, recall is producer's accuracy, F1 is their harmonic mean, and intersection-over-union is TP/(TP+FP+FN). Cohen's kappa adjusts observed agreement for agreement expected from class marginals. Fixed NDWI/MNDWI thresholds and Otsu thresholds are scored on the same held-out points as the tree. A block bootstrap resamples complete spatial blocks for confidence intervals, preserving within-block dependence.

The primary saved run's 98.52% overall accuracy is agreement with JRC-derived labels on held-out blocks. It is not a field-survey accuracy estimate. The same-sample MNDWI Otsu baseline is 98.79%; this comparison is explicitly retained because a learned estimator need not outperform a threshold on every dataset.

## Seasonal transfer and area/change calculations

The tree is trained on the configured dry-season composite, then applied to each valid pixel in each selected period. Output classes use 0=land, 1=water, and 255=NoData. Valid water area is pixel count multiplied by projected pixel area. Change classes are calculated only where both seasonal rasters are valid: 0=land in both, 1=water in both, 2=land-to-water gain, 3=water-to-land loss. Net change is gain minus loss; area differences across different valid coverage are not treated as change.

The Streamlit app reads the prebuilt rasters, image previews, summary JSON, and tree diagnostics from `app/data/`. It does not authenticate with Earth Engine, retrain the tree, or rebuild the maps at page load. `src/waterdet/presentation.py` creates these display assets and `src/waterdet/change.py` handles aligned common-coverage comparison.

## Reproducibility and test vocabulary

- **Real-data run:** the recorded Lagos products come from exported Sentinel-2 imagery and JRC labels; hashes and run parameters are recorded in the summary and review artifacts.
- **Synthetic fixture:** generated rasters with known truth used only by tests/demo to test the complete processing path and edge cases. They are not evidence for Lagos performance.
- **Offline test:** Earth Engine calls are mocked or avoided; local numerical and raster behavior can be validated without Google credentials.
- **End-to-end test:** runs the raster pipeline and presentation/app asset builder on controlled fixture data, then asserts outputs and known-truth behavior.

Run `python -m pytest -q` for the same test command used in `.github/workflows/tests.yml`. The suite covers raster IO and alignment, indices, label semantics, split leakage, model fitting, metrics and reference weighting, change classes, presentation packaging, and Streamlit rendering/interactions.
