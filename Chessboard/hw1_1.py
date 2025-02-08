import cv2
import numpy as np

# Load the input image
image_path = "checker.png"
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image_path}")

# Resize the image to a manageable size
image = cv2.resize(image, (600, 600))

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Use Canny edge detection to detect edges
edges = cv2.Canny(blurred, 15, 75)  # Lower thresholds for Canny to detect more edges

# Use morphological operations to close gaps in edges
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# Find contours in the closed image
contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours to find squares that are likely to be the checkerboard cells
checkerboard_contours = []
for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.05 * cv2.arcLength(contour, True), True)  # Increase approximation factor to relax criteria
    area = cv2.contourArea(contour)
    if len(approx) == 4 and 300 < area < 60000:  # Lower area thresholds to detect smaller cells
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        if 0.6 < aspect_ratio < 1.4:  # Broaden aspect ratio range
            checkerboard_contours.append(contour)

# Sort contours to find the grid layout by their position
checkerboard_contours = sorted(checkerboard_contours, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))

# Count the number of detected cells
num_cells = len(checkerboard_contours)

# Determine the board size based on the number of detected cells
if 71 <= num_cells <= 110:
    print("국제 룰: 10 x 10 크기")
elif 40 <= num_cells <= 70:
    print("영/미식 룰: 8 x 8 크기")
else:
    print("알 수 없는 크기: 감지된 셀의 수 = {}".format(num_cells))