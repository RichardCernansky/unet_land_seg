import cv2
import numpy as np


def closing_opening_to_file(src_path, dst_path, gap_diameter_px, reach_diameter_px):
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    _, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY) # discarding original binary
    if gap_diameter_px > 0:
        r = int(np.ceil(gap_diameter_px/2))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=1)
    if reach_diameter_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (reach_diameter_px+1, reach_diameter_px+1))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)
    cv2.imwrite(dst_path, bw)
    

gap_diameter_px = 200
reach_diameter_px = 53
closing_opening_to_file("./data/predicted_masks/Buriny.png", "./data/predicted_masks/Buriny_post.png", gap_diameter_px, reach_diameter_px)


