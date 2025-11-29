import os

# --- Basic Settings ---
DEVICE = "cuda"
MODEL_PATH = os.path.join('weights', 'dceunetex.pth')

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 512

# --- Fusion & Refinement Parameters ---
# Median filter kernel sizes (Must be odd)
MEDIAN_FILTER_SIZES = [1, 5, 7] 

# Weights for fusing edges from different scales
CANNY_FUSION_WEIGHTS = [0.5, 0.3, 0.2]

# Spatial Confidence Map Sigmas
CONFIDENCE_MAP_SIGMAS = [10, 15, 20]

# Weights for fusing confidence maps
CONFIDENCE_MAP_WEIGHTS = [0.5, 0.3, 0.2]

# Threshold for generating candidate points for RANSAC
FUSED_MAP_FINAL_THRESHOLD = 60

# Hough Linearity Filter Parameters
HOUGH_THRESHOLD = 50       
HOUGH_MIN_LINE_LENGTH = 60 
HOUGH_MAX_LINE_GAP = 100