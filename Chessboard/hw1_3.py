import cv2
import numpy as np

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# Load the image
image = cv2.imread('checker.png')

if image is None:
    print("Error: Could not load image. Please check the file path.")
    exit()

# Resize image to a smaller size for easier processing (optional)
image = cv2.resize(image, (800, 800))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Use adaptive thresholding for better results
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

board_contour = None
for contour in contours:
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) == 4:
        area = cv2.contourArea(approx)
        if area > 10000:  # Ignore small contours
            board_contour = approx
            break

if board_contour is None:
    print("Error: Could not find a valid board contour.")
    exit()

# Perspective transformation
pts = board_contour.reshape(4, 2)
rect = order_points(pts)

(tl, tr, br, bl) = rect
width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
max_width = max(int(width_a), int(width_b))

height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
max_height = max(int(height_a), int(height_b))

max_dimension = int(max(max_width, max_height) * 1.1)


# Ensure the final output is a square
dst = np.array([
    [0, 0],
    [max_dimension - 1, 0],
    [max_dimension - 1, max_dimension - 1],
    [0, max_dimension - 1]
], dtype="float32")

M = cv2.getPerspectiveTransform(rect, dst)
warped = cv2.warpPerspective(image, M, (max_dimension, max_dimension))

# Convert to grayscale and use binary threshold to focus on board grid
warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
_, warped_thresh = cv2.threshold(warped_gray, 150, 255, cv2.THRESH_BINARY)

# Find and sort the contours of the squares on the chessboard
contours, _ = cv2.findContours(warped_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
squares = [c for c in contours if 500 < cv2.contourArea(c) < 10000]

# Sort squares by their position to ensure a logical order (left to right, top to bottom)
squares = sorted(squares, key=lambda c: (cv2.boundingRect(c)[1] // (max_dimension // 8), cv2.boundingRect(c)[0]))

# Create a mask and draw all the detected squares
mask = np.zeros_like(warped_gray)
for square in squares:
    x, y, w, h = cv2.boundingRect(square)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

# Invert the mask to keep the original image pixels outside of the squares
mask_inv = cv2.bitwise_not(mask)

# Extract the original image parts where the mask is not applied
background = cv2.bitwise_and(warped, warped, mask=mask_inv)

# Extract the squares from the warped image
cropped_warped = cv2.bitwise_and(warped, warped, mask=mask)

# Combine both parts to retain the original image outside the squares and the squares themselves
final_result = cv2.add(background, cropped_warped)

# Get bounding box of all the detected squares to crop the final output
x_coords = [cv2.boundingRect(square)[0] for square in squares]
y_coords = [cv2.boundingRect(square)[1] for square in squares]
x_min, x_max = min(x_coords) - 5, max(x_coords) + w 
y_min, y_max = min(y_coords) - 5, max(y_coords) + h 
final_cropped = final_result[y_min:y_max, x_min:x_max]

# Resize the final cropped image to be a square
final_cropped_square = cv2.resize(final_cropped, (max_dimension, max_dimension))

cv2.imshow('Warped Image', final_cropped_square)
cv2.waitKey(0)
cv2.destroyAllWindows()
