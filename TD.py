import numpy as np
import rasterio
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from decimal import Decimal, getcontext

# Set precision for decimal arithmetic
getcontext().prec = 50

# File paths for ground truth and predicted data
ground_truth_path = r"C:\Users\Mahsa\OneDrive\Desktop\gt_new.tif"
model_result_path = r"C:\Users\Mahsa\OneDrive\Desktop\result.tif"

# Load ground truth and predicted data
with rasterio.open(ground_truth_path) as src:
    ground_truth = src.read(1)
    target_shape = ground_truth.shape

with rasterio.open(model_result_path) as src:
    model_result = src.read(1)

# Ensure matching dimensions
assert ground_truth.shape == model_result.shape, "Dimensions of ground truth and predicted data do not match!"

# Flatten arrays for confusion matrix calculation
ground_truth_flat = ground_truth.flatten()
model_result_flat = model_result.flatten()

# Define the class labels
# Post-earthquake building damage assessment classes
# 0: Background (no building), 1: No Damage, 2: Minor Damage, 3: Major Damage, 4: Destroyed
class_labels = [0, 1, 2, 3, 4]

CLASS_NAMES = {
    0: "Background",
    1: "No Damage",
    2: "Minor Damage",
    3: "Major Damage",
    4: "Destroyed",
}

# Calculate confusion matrix
conf_matrix = confusion_matrix(ground_truth_flat, model_result_flat, labels=class_labels)

# Initialize dictionaries for class matrices and metrics
class_matrices = {}
metrics = {}

# Compute 2x2 matrix and metrics for each class
for i, label in enumerate(class_labels):
    tp = conf_matrix[i, i]  # True Positive
    fn = conf_matrix[i, :].sum() - tp  # False Negative
    fp = conf_matrix[:, i].sum() - tp  # False Positive
    tn = conf_matrix.sum() - (tp + fn + fp)  # True Negative

    # Store the 2x2 matrix for this class
    class_matrices[label] = np.array([[tp, fn], [fp, tn]])

    # Calculate performance metrics for the current class
    TPR = tp / (tp + fn) if (tp + fn) != 0 else 0  # Sensitivity/Recall
    SPC = tn / (tn + fp) if (tn + fp) != 0 else 0  # Specificity
    PPV = tp / (tp + fp) if (tp + fp) != 0 else 0  # Precision
    NPV = tn / (tn + fn) if (tn + fn) != 0 else 0  # Negative Predictive Value
    FPR = fp / (fp + tn) if (fp + tn) != 0 else 0  # False Positive Rate
    FNR = fn / (fn + tp) if (fn + tp) != 0 else 0  # False Negative Rate
    ACC = (tp + tn) / (tp + fp + fn + tn)  # Accuracy
    F1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0  # F1 Score
    F_Measure = (2 * TPR * PPV) / (TPR + PPV) if (TPR + PPV) != 0 else 0  # F-Measure
    Informedness = TPR + SPC - 1  # Informedness
    Markedness = PPV + NPV - 1  # Markedness
    IoU = tp / (tp + fp + fn) if (tp + fp + fn) != 0 else 0  # Intersection over Union
    mIoU = IoU  # Mean IoU

    # Convert numpy int64 to Python int before creating a Decimal
    tp_d = Decimal(int(tp))
    tn_d = Decimal(int(tn))
    fp_d = Decimal(int(fp))
    fn_d = Decimal(int(fn))

    # Matthews Correlation Coefficient (MCC)
    mcc_numerator = (tp_d * tn_d) - (fp_d * fn_d)
    mcc_denominator = (tp_d + fp_d) * (tp_d + fn_d) * (tn_d + fp_d) * (tn_d + fn_d)
    mcc_denominator_root = mcc_denominator.sqrt() if mcc_denominator > 0 else Decimal(0)
    MCC = mcc_numerator / mcc_denominator_root if mcc_denominator_root != 0 else Decimal('nan')

    # Store the metrics for this class
    metrics[label] = {
        "TPR": TPR,
        "SPC": SPC,
        "PPV": PPV,
        "NPV": NPV,
        "FPR": FPR,
        "FNR": FNR,
        "ACC": ACC,
        "F1": F1,
        "F_Measure": F_Measure,
        "Informedness": Informedness,
        "Markedness": Markedness,
        "IoU": IoU,
        "mIoU": mIoU,
        "MCC": float(MCC)  # Convert back to float for display
    }

# Print the 2x2 matrices and metrics for each class
for label, matrix in class_matrices.items():
    print(f"Class {label}: {CLASS_NAMES[label]}")
    print(matrix)
    print(f"Metrics for Class {label}:")
    for metric, value in metrics[label].items():
        print(f"{metric}: {value}")
    print()

# Compute overall mIoU across all classes
overall_mIoU = np.mean([metrics[label]["IoU"] for label in class_labels])
print(f"Overall mIoU (mean over all classes): {overall_mIoU}")

# Create binary images for visualization
# Binary = "any damage class present" (1-4) vs. background (0)
ground_truth_binary = (ground_truth >= 1).astype(np.uint8)
model_result_binary = (model_result >= 1).astype(np.uint8)

# Create RGB image for visualization
rgb_image = np.zeros((ground_truth.shape[0], ground_truth.shape[1], 3), dtype=np.uint8)

rgb_image[(ground_truth_binary == 1) & (model_result_binary == 1)] = [255, 255, 255]  # True Positive (white)
rgb_image[(ground_truth_binary == 0) & (model_result_binary == 1)] = [0, 0, 255]  # False Positive (blue)
rgb_image[(ground_truth_binary == 0) & (model_result_binary == 0)] = [255, 255, 255]  # True Negative (white)
rgb_image[(ground_truth_binary == 1) & (model_result_binary == 0)] = [255, 0, 0]  # False Negative (red)

# Display the RGB visualization
plt.figure(figsize=(12, 12))
plt.imshow(rgb_image)
plt.title('Building Damage Detection Results')

handles = [
    mpatches.Patch(color='white', label='TN & TP (White)'),
    mpatches.Patch(color='blue', label='FP (Blue)'),
    mpatches.Patch(color='red', label='FN (Red)'),
]
plt.legend(handles=handles, loc='lower left', fontsize=6)
plt.axis('off')
plt.show()
