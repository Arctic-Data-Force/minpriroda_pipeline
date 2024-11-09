import logging
import os

import cv2
import pandas as pd
from tqdm import tqdm

# Path to the images folder
images_folder = 'train_data_minprirodi/images'
def normalized_bbox_to_pixel(bbox, image_width, image_height):
    """
    Convert normalized bbox [x_center, y_center, width, height] to pixel coordinates [x1, y1, x2, y2].
    """
    x_center, y_center, width, height = map(float, bbox.split(','))
    x_center *= image_width
    y_center *= image_height
    width *= image_width
    height *= image_height

    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    return [x1, y1, x2, y2]

def calculate_iou(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Each box is a list of four coordinates: [x1, y1, x2, y2].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) != 0 else 0

    return iou

# Paths to the ground truth and submission CSV files
ground_truth_path = 'train_data_minprirodi/annotation.csv'
submission_path = 'submission.csv'

# Load the CSV files
ground_truth_df = pd.read_csv(ground_truth_path)
submission_df = pd.read_csv(submission_path)

# Preview the data
print("Ground Truth Sample:")
print(ground_truth_df.head())

print("\nSubmission Sample:")
print(submission_df.head())

# Group ground truth by image
gt_grouped = ground_truth_df.groupby('Name')

# Group submission by image
submission_grouped = submission_df.groupby('Name')

# Initialize scores
detector_score = 0
classifier_score = 0

# Counters for total objects
total_gt_objects = 0
total_submission_objects = 0

# Additional Counters for Detected Metrics
detector_tp = 0  # True Positives
detector_fp = 0  # False Positives
detector_fn = 0  # False Negatives

# Additional Counters for Classification Metrics
classifier_cc = 0  # Correct Classifications
classifier_ic = 0  # Incorrect Classifications
classifier_fpc = 0  # False Positive Classifications

# Iterate over each image in ground truth
for image_name, gt_rows in tqdm(gt_grouped, desc="Calculating Detector Quality"):
    # Load the image to get dimensions
    image_path = os.path.join(images_folder, image_name)
    img = cv2.imread(image_path)

    if img is None:
        logging.warning(f"Unable to read image {image_name} for evaluation. Skipping.")
        continue

    img_height, img_width = img.shape[:2]

    # Get ground truth bboxes
    gt_bboxes = gt_rows['Bbox'].tolist()
    gt_bboxes_pixel = [normalized_bbox_to_pixel(bbox, img_width, img_height) for bbox in gt_bboxes]

    # Get submission bboxes for this image
    if image_name in submission_grouped.groups:
        sub_rows = submission_grouped.get_group(image_name)
        sub_bboxes = sub_rows['Bbox'].tolist()
        sub_bboxes_pixel = [normalized_bbox_to_pixel(bbox, img_width, img_height) for bbox in sub_bboxes]
        sub_classes = sub_rows['Class'].tolist()
    else:
        sub_bboxes_pixel = []
        sub_classes = []

    total_gt_objects += len(gt_bboxes_pixel)
    total_submission_objects += len(sub_bboxes_pixel)

    # Keep track of matched ground truth
    matched_gt = set()

    # Iterate over submission bboxes and match with ground truth
    for sub_box in sub_bboxes_pixel:
        best_iou = 0
        best_gt_idx = -1
        for gt_idx, gt_box in enumerate(gt_bboxes_pixel):
            if gt_idx in matched_gt:
                continue
            iou = calculate_iou(sub_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        if best_iou > 0.5:
            detector_score += 1  # Correct detection
            detector_tp += 1
            matched_gt.add(best_gt_idx)
        else:
            detector_score -= 1  # False positive
            detector_fp += 1

    # Penalize for missed ground truth objects
    missed_objects = len(gt_bboxes_pixel) - len(matched_gt)
    detector_score -= missed_objects
    detector_fn += missed_objects

# Reset matched_gt for classifier evaluation
for image_name, gt_rows in tqdm(gt_grouped, desc="Calculating Classifier Quality"):
    # Load the image to get dimensions
    image_path = os.path.join(images_folder, image_name)
    img = cv2.imread(image_path)

    if img is None:
        continue  # Already logged in previous loop

    img_height, img_width = img.shape[:2]

    # Get ground truth bboxes and classes
    gt_bboxes = gt_rows['Bbox'].tolist()
    gt_bboxes_pixel = [normalized_bbox_to_pixel(bbox, img_width, img_height) for bbox in gt_bboxes]
    gt_classes = gt_rows['Class'].tolist()

    # Get submission bboxes and classes for this image
    if image_name in submission_grouped.groups:
        sub_rows = submission_grouped.get_group(image_name)
        sub_bboxes = sub_rows['Bbox'].tolist()
        sub_bboxes_pixel = [normalized_bbox_to_pixel(bbox, img_width, img_height) for bbox in sub_bboxes]
        sub_classes = sub_rows['Class'].tolist()
    else:
        sub_bboxes_pixel = []
        sub_classes = []

    # Keep track of matched ground truth
    matched_gt = set()

    # Iterate over submission bboxes and match with ground truth
    for sub_box, sub_class in zip(sub_bboxes_pixel, sub_classes):
        best_iou = 0
        best_gt_idx = -1
        for gt_idx, gt_box in enumerate(gt_bboxes_pixel):
            if gt_idx in matched_gt:
                continue
            iou = calculate_iou(sub_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        if best_iou > 0.5:
            # Correct detection
            matched_gt.add(best_gt_idx)
            # Check if class matches
            if sub_class == gt_classes[best_gt_idx]:
                classifier_score += 5  # Correct class
                classifier_cc += 1
            else:
                classifier_score -= 5  # Incorrect class
                classifier_ic += 1
        else:
            # False positive in classification
            classifier_score -= 5
            classifier_fpc += 1

    # Penalize for missed ground truth objects (no class to assign, only detector)
    # Already handled in detector_score


# Calculate total possible scores
# Assuming M is the number of auxiliary objects (not defined in the original description)
# Here, we'll consider M = 0 for simplicity. Adjust accordingly if you have M.
M = 0
N = total_gt_objects  # Number of ground truth objects

# Maximum possible score
max_score = (M + N) * 6

# Sum of detector and classifier scores
X = detector_score + classifier_score

# Final result
final_result = X
if X > 0:
    final_result = X / max_score
elif X <= 0:
    final_result = 0

# Ensure the final result is not negative
final_result = max(final_result, 0)

print(f"\n--- Итоговые Метрики ---")
print(f"Всего объектов в разметке (GT): {N}")
print(f"Всего обнаруженных объектов (Submission): {total_submission_objects}\n")

print(f"Детектор:")
print(f"  True Positives (TP): {detector_tp}")
print(f"  False Positives (FP): {detector_fp}")
print(f"  False Negatives (FN): {detector_fn}\n")

print(f"Классификатор:")
print(f"  Correct Classifications (CC): {classifier_cc}")
print(f"  Incorrect Classifications (IC): {classifier_ic}")
print(f"  False Positive Classifications (FPC): {classifier_fpc}\n")


print(f"Итоговый Балл: {final_result:.2f}% {X} {max_score}")
