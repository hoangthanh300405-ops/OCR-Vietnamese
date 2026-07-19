import cv2
import numpy as np


def Deskew_image(image):
    h, w = image.shape[:2]
    kernel_w = max(5, w // 4)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 5))
    dilate = cv2.dilate(image, kernel, iterations=1)

    pts = cv2.findNonZero(dilate)
    if pts is None or pts.shape[0] < 10:
        return image

    n_pts = pts.shape[0]
    if n_pts > 1000:
        idx = np.random.choice(n_pts, 1000, replace=False)
        pts = pts[idx]

    rect = cv2.minAreaRect(pts)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    elif angle > 45:
        angle = 90 - angle
    else:
        angle = -angle

    if abs(angle) < 0.5:
        return image

    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return rotated


def preprocess_pipeline(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    sharpen_kernel = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])
    gray = cv2.filter2D(gray, -1, sharpen_kernel)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    binary_adaptive = cv2.adaptiveThreshold(gray, 255,
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
    _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    binary = cv2.bitwise_or(binary_adaptive, binary_otsu)

    inv = cv2.bitwise_not(binary)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    kernel_med = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    inv = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel_small, iterations=1)
    inv = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel_med, iterations=1)

    binary = cv2.bitwise_not(inv)

    return binary


def distance_transform(binary):
    inv = cv2.bitwise_not(binary)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return dist


def pre_processing(image):
    binary = preprocess_pipeline(image)
    binary = Deskew_image(binary)
    binary = cv2.resize(binary, (128, 256), interpolation=cv2.INTER_NEAREST)
    binary = distance_transform(binary)
    return binary
