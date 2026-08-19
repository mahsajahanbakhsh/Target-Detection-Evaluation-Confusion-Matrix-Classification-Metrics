# Target-Detection-Evaluation-Confusion-Matrix-Classification-Metrics
# Post-Earthquake Building Damage Detection — Confusion Matrix & Classification Metrics

---

## Assignment Description

Given two raster outputs — a **model damage-detection result** and its corresponding **ground truth** — this assignment:

1. Builds the confusion matrix and displays it.
2. Computes TPR, SPC, PPV, NPV, FPR, FNR, ACC, F1 Score, MCC, and F-measure for each class.
3. Visualizes ground truth vs. detected results, color-coding Type I and Type II errors according to the class conventions.
4. Computes and reports IoU and mIoU.

---

## Project Overview

This project evaluates a pixel-wise **post-earthquake building damage assessment** model against ground truth using raster (GeoTIFF) data. Two GeoTIFF rasters — model predictions and ground truth — are compared pixel-by-pixel across **5 damage classes** over a residential building footprint area, producing a full multi-class confusion matrix, per-class 2×2 confusion breakdowns, a comprehensive set of classification metrics, a color-coded error map, and (optionally) styled summary tables.

The classification map (visible in the detection-results figure) clearly shows building outlines/footprints across a residential block, confirming this is a **building damage assessment** task rather than a land-cover/forest classification task.

---

## Data

| File | Description |
|---|---|
| `gt_new.tif` | Ground truth raster (damage class label per pixel) |
| `result.tif` | Model detection result raster (predicted damage class label per pixel) |

**Classes:**

| Label | Class |
|---|---|
| 0 | Background (no building) |
| 1 | No Damage |
| 2 | Minor Damage |
| 3 | Major Damage |
| 4 | Destroyed |

This follows a standard post-disaster building damage severity scale, ranging from intact structures to full collapse.

---

## Pipeline Structure

1. **Data Loading** — read both rasters with `rasterio`, verify matching dimensions
2. **Confusion Matrix** — compute the 5×5 multi-class confusion matrix with `sklearn.metrics.confusion_matrix`
3. **Per-Class 2×2 Breakdown** — for each damage class, derive TP, FN, FP, TN from the multi-class matrix (one-vs-rest)
4. **Metric Computation** — for each class, calculate:
   - **TPR** (Sensitivity/Recall), **SPC** (Specificity), **PPV** (Precision), **NPV** (Negative Predictive Value)
   - **FPR** (False Positive Rate), **FNR** (False Negative Rate)
   - **ACC** (Accuracy), **F1 Score**, **F-Measure**
   - **Informedness**, **Markedness**
   - **IoU** (per class) and overall **mIoU** (mean IoU across all 5 classes)
   - **MCC** (Matthews Correlation Coefficient), computed with high-precision decimal arithmetic to avoid overflow/precision loss on large pixel counts
5. **Error Visualization** — binarize both rasters (any damage class ≥ 1 = "building/damage present") and render an RGB error map:
   - **White** — True Negative & True Positive (correct)
   - **Blue** — False Positive (Type I error — damage/building detected where none exists)
   - **Red** — False Negative (Type II error — missed damage/building)

---

## Results

The figure shows building footprints across a residential block, with **blue regions marking false positives** (over-detected damage/buildings) and **red regions marking false negatives** (missed damage/buildings). The bulk of the map is white, indicating the model correctly agrees with ground truth for most of the area — errors are concentrated around individual building edges and a cluster of false positives in the upper-right of the scene.

The underlying pixel counts and computed metrics are unchanged from the original run (only the class labels were corrected):

### Confusion Counts per Class

| | Background | No Damage | Minor Damage | Major Damage | Destroyed |
|---|---|---|---|---|---|
| **TP** | 23,567,862 | 3,170,629 | 33,083 | 55,318 | 186,298 |
| **FP** | 164,942 | 304,056 | 12,111 | 6,733 | 24,088 |
| **FN** | 281,512 | 140,835 | 41,082 | 13,406 | 35,095 |
| **TN** | 3,510,804 | 23,909,600 | 27,438,844 | 27,449,663 | 27,279,639 |

### Classification Metrics per Class

| Metric | Background | No Damage | Minor Damage | Major Damage | Destroyed |
|---|---|---|---|---|---|
| TPR | 0.9882 | 0.9575 | 0.4461 | 0.8049 | 0.8415 |
| SPC | 0.9551 | 0.9874 | 0.9996 | 0.9998 | 0.9991 |
| PPV | 0.9931 | 0.9125 | 0.7320 | 0.8915 | 0.8855 |
| NPV | 0.9258 | 0.9941 | 0.9985 | 0.9995 | 0.9987 |
| FPR | 0.0449 | 0.0126 | 0.0004 | 0.0002 | 0.0009 |
| FNR | 0.0118 | 0.0425 | 0.5539 | 0.1951 | 0.1585 |
| ACC | 0.9838 | 0.9838 | 0.9981 | 0.9993 | 0.9978 |
| F1 Score | 0.9906 | 0.9344 | 0.5543 | 0.8460 | 0.8629 |
| F-Measure | 0.9906 | 0.9344 | 0.5543 | 0.8460 | 0.8629 |
| Informedness | 0.9433 | 0.9449 | 0.4456 | 0.8047 | 0.8406 |
| Markedness | 0.9188 | 0.9066 | 0.7305 | 0.8910 | 0.8842 |
| **IoU** | **0.9814** | **0.8770** | **0.3835** | **0.7331** | **0.7589** |
| MCC | 0.9310 | 0.9256 | 0.5706 | 0.8467 | 0.8621 |

### Mean IoU (mIoU)

Averaging the per-class IoU across all 5 classes:

**mIoU = 0.7468**

> **Background** and **No Damage** are detected very reliably (IoU > 0.87) — the model is strong at telling "building present, undamaged" apart from "no building." **Minor Damage**, however, is the weakest class (IoU = 0.38, TPR = 0.45) — more than half of true minor-damage pixels are missed (FNR = 0.55), most likely because minor damage sits on a visually ambiguous boundary between no-damage and major-damage, making it the hardest severity level to separate. **Major Damage** and **Destroyed** fall in between, with solid agreement (IoU ≈ 0.73–0.76), meaning the model is fairly reliable at flagging severe damage even if it sometimes confuses the exact severity tier.

---

## Requirements

```
numpy
pandas
matplotlib
rasterio
scikit-learn
```

Quick install:

```bash
pip install numpy pandas matplotlib rasterio scikit-learn
```

---

## How to Run

1. Update `ground_truth_path` and `model_result_path` in `TD2_corrected.py` to point to your `gt_new.tif` and `result.tif` files.
2. Run the script to:
   - Compute the confusion matrix and per-class metrics for all 5 damage classes
   - Print the 2×2 breakdown and metrics for each class, correctly labeled (Background / No Damage / Minor Damage / Major Damage / Destroyed)
   - Print the overall mIoU across all classes
   - Generate the RGB damage-detection error visualization

---

## Notes

- The multi-class confusion matrix is reduced to 5 independent one-vs-rest 2×2 matrices per class; this is standard for multi-class TPR/PPV/IoU reporting but means the reported per-class TN counts include pixels from *all other classes*, not a single "negative" class.
- MCC is computed with Python's `decimal` module at 50-digit precision to avoid floating-point overflow when multiplying very large pixel counts (tens of millions) — a good practice given the scale of this raster.
- mIoU is a simple unweighted mean across the 5 classes; if class frequency should be taken into account, a frequency-weighted IoU could be reported alongside it.
- **Class naming correction:** the class labels in this project describe **building damage severity** (Background / No Damage / Minor Damage / Major Damage / Destroyed), not forest cover types. Any earlier report or notebook referring to "Dense Forest" / "Non-dense Forest" / "Non-Forest" for this dataset used the wrong labels for this task and should be disregarded in favor of the damage-class scheme above.
