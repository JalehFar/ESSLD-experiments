import cv2
import numpy as np
import math
import numba
import config

@numba.jit(nopython=True)
def _ray_cast_core_numba(contour_map, p_x, p_y, max_dist, num_rays, h, w):
    out_points = np.zeros((num_rays, 2), dtype=np.int32)
    count = 0
    
    for i in range(num_rays):
        angle = 2 * math.pi * i / num_rays
        dir_x = math.cos(angle)
        dir_y = math.sin(angle)
        
        found = False
        for r in range(1, max_dist):
            curr_x = int(p_x + r * dir_x)
            curr_y = int(p_y + r * dir_y)
            
            if curr_x < 0 or curr_x >= w or curr_y < 0 or curr_y >= h:
                break
            
            if contour_map[curr_y, curr_x] > 0:
                out_points[count, 0] = curr_x
                out_points[count, 1] = curr_y
                count += 1
                found = True
                break
                
    return out_points[:count]

@numba.jit(nopython=True)
def _non_max_suppression_fast(magnitude, angle):
    H, W = magnitude.shape
    suppressed = np.zeros_like(magnitude)
    angle = angle * 180. / np.pi
    for i in range(H):
        for j in range(W):
            if angle[i, j] < 0: angle[i, j] += 180
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            q, r = 255.0, 255.0
            a = angle[i, j]
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                q, r = magnitude[i, j+1], magnitude[i, j-1]
            elif (22.5 <= a < 67.5):
                q, r = magnitude[i+1, j-1], magnitude[i-1, j+1]
            elif (67.5 <= a < 112.5):
                q, r = magnitude[i+1, j], magnitude[i-1, j]
            elif (112.5 <= a < 157.5):
                q, r = magnitude[i-1, j-1], magnitude[i+1, j+1]
            if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                suppressed[i, j] = magnitude[i, j]
            else:
                suppressed[i, j] = 0
    return suppressed

def _robust_fit_line(points, dist_type=cv2.DIST_HUBER):
    if len(points) < 2: return None, None
    points = np.array(points, dtype=np.float32)
    
    line = cv2.fitLine(points, dist_type, 0, 0.01, 0.01)
    
    line = line.flatten() 
    vx, vy, x0, y0 = line[0], line[1], line[2], line[3]

    if abs(vx) < 1e-5: 
        vx = 1e-5 
    
    k = vy / vx
    b = y0 - k * x0
 
    return float(k), float(b)

def _is_gourd_shape(mask, threshold=0.8):
    small_h, small_w = mask.shape[0] // 4, mask.shape[1] // 4
    if small_h == 0 or small_w == 0: return False
    
    small_mask = cv2.resize(mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    
    contours, _ = cv2.findContours(small_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return False
    largest_contour = max(contours, key=cv2.contourArea)
    if len(largest_contour) < 5: return False
    
    rect = cv2.minAreaRect(largest_contour)
    (_, _), (width, height), angle = rect
    if width < height: angle += 90
    
    rot_mat = cv2.getRotationMatrix2D((small_w / 2, small_h / 2), angle, 1.0)
    rotated_mask = cv2.warpAffine(small_mask, rot_mat, (small_w, small_h))
    
    horizontal_projection = np.sum(rotated_mask, axis=1) / 255.0
    non_zero_projection = horizontal_projection[horizontal_projection > 0]
    
    if len(non_zero_projection) < 3: return False
    min_width, max_width = np.min(non_zero_projection), np.max(non_zero_projection)
    return min_width < max_width * threshold

def _find_optimal_p(mask, bounding_box):
    x, y, w, h = bounding_box
    search_mask = mask.copy()
    if _is_gourd_shape(mask):
        search_mask = np.zeros_like(mask)
        upper_y_end = y + h // 2
        search_mask[y:upper_y_end, x:x + w] = mask[y:upper_y_end, x:x + w]
        if np.sum(search_mask) == 0: search_mask = mask
    
    if np.sum(search_mask) == 0: return None
    dist_map = cv2.distanceTransform(search_mask, cv2.DIST_L2, 5)
    _, _, _, max_loc = cv2.minMaxLoc(dist_map)
    return max_loc

def _ray_cast_to_contour(p_point, contour, mask_shape, num_rays):
    h, w = mask_shape
    contour_image = np.zeros(mask_shape, dtype=np.uint8)
    cv2.drawContours(contour_image, [contour], -1, 255, 1)
    
    max_dist = int(math.sqrt(h ** 2 + w ** 2))
 
    raw_points = _ray_cast_core_numba(contour_image, p_point[0], p_point[1], max_dist, num_rays, h, w)
    
    if len(raw_points) == 0: return None
    
    return raw_points.reshape(-1, 1, 2)

def get_coarse_line_from_mask(mask, original_frame):
    h, w = mask.shape[:2]
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
    if num_labels < 2: return None

    largest_label = 1
    max_area = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > max_area:
            max_area = stats[i, cv2.CC_STAT_AREA]
            largest_label = i
            
    if max_area < 100: return None 

    clean_mask = np.zeros_like(mask)
    clean_mask[labels == largest_label] = 255

    repaired_mask = np.zeros_like(clean_mask)
    try:
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(largest_contour)
            cv2.fillPoly(repaired_mask, [hull], 255)
        else:
            repaired_mask = clean_mask
    except Exception:
        repaired_mask = clean_mask

    horizon_points = []
    mask_int16 = repaired_mask.astype(np.int16) 
    
    for col in range(0, w, 4):
        column = mask_int16[:, col]
        diff = np.diff(column)
        transitions = np.where(np.abs(diff) > 100)[0]
        
        if len(transitions) > 0:
            y_pos = transitions[0]
            if 5 < y_pos < h - 5:
                horizon_points.append([col, y_pos])
    
    if len(horizon_points) < 10: return None
    points = np.array(horizon_points)
    
    try:
        k, b = _robust_fit_line(points, cv2.DIST_HUBER)
        if k is None: return None

        pred_y = k * points[:, 0] + b
        diff = np.abs(points[:, 1] - pred_y)
        inliers = points[diff < 5.0]
        
        if len(inliers) > 10:
            k_final, b_final = np.polyfit(inliers[:, 0], inliers[:, 1], 1)
            angle = np.rad2deg(np.arctan(k_final))
            y_mid = k_final * (w / 2) + b_final
            return y_mid, angle
        
        angle = np.rad2deg(np.arctan(k))
        y_mid = k * (w / 2) + b
        return y_mid, angle

    except Exception:
        return None

#  FUSION CORE

def _run_dual_fusion_pipeline(roi_image):
    if roi_image is None: return None
    h, w = roi_image.shape[:2]

    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    edge_accum = np.zeros_like(gray, dtype=np.float32)
    for k, weight in zip(config.MEDIAN_FILTER_SIZES, config.CANNY_FUSION_WEIGHTS):
        blurred = cv2.medianBlur(gray, k)
        edge = cv2.Canny(blurred, 50, 150)
        edge_accum += (edge / 255.0) * weight
    edge_norm = cv2.normalize(edge_accum, None, 0, 1, cv2.NORM_MINMAX)

    center_y = h // 2
    y_grid = np.arange(h).reshape(-1, 1)
    dist_map = np.abs(y_grid - center_y)
    
    conf_accum = np.zeros_like(gray, dtype=np.float32)
    for sigma, weight in zip(config.CONFIDENCE_MAP_SIGMAS, config.CONFIDENCE_MAP_WEIGHTS):
        conf = np.exp(-0.5 * (dist_map / sigma)**2)
        conf = np.tile(conf, (1, w))
        conf_accum += (edge_norm * conf) * weight

    sobelx = cv2.Sobel(conf_accum, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(conf_accum, cv2.CV_64F, 0, 1, ksize=5)
    angle = np.arctan2(sobely, sobelx)
    nms_map = _non_max_suppression_fast(conf_accum, angle)
    
    binary_map = (nms_map * 255 > config.FUSED_MAP_FINAL_THRESHOLD).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_map, 4, cv2.CV_32S)
    if num_labels > 1:
        cleaned_map = np.zeros_like(binary_map)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= 50:
                cleaned_map[labels == i] = 255
        binary_map = cleaned_map

    lines = cv2.HoughLinesP(binary_map, 1, np.pi/180, config.HOUGH_THRESHOLD, 
                            config.HOUGH_MIN_LINE_LENGTH, config.HOUGH_MAX_LINE_GAP)
    if lines is not None:
        mask = np.zeros_like(binary_map)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(mask, (x1, y1), (x2, y2), 255, 1)
        binary_map = cv2.bitwise_and(binary_map, binary_map, mask=mask)

    points = np.argwhere(binary_map > 0)
    if len(points) < 10: return None
    
    pts_xy = np.column_stack((points[:, 1], points[:, 0]))

    try:
        weights = conf_accum[points[:, 0], points[:, 1]]
        
        if np.sum(weights) > 1e-6:
             coeffs = np.polyfit(pts_xy[:, 0], pts_xy[:, 1], 1, w=weights)
             k, b = coeffs[0], coeffs[1]
        else:
             k, b = _robust_fit_line(pts_xy, cv2.DIST_L12)
        
        if k is None: return None
        
        angle = np.rad2deg(np.arctan(k))
        y_mid = k * (w / 2) + b
        return y_mid, angle 
    except: 
        return None

def _calculate_dynamic_padding(grad_score):
    norm_score = (grad_score - 25.0) / (200.0 - 25.0)
    norm_score = np.clip(norm_score, 0, 1)
    padding = 40 - norm_score * (40 - 20)
    return int(padding)

def _create_roi(frame, y_coarse, angle_coarse, padding):
    h, w = frame.shape[:2]
    roi_center_y = y_coarse
    roi_angle = angle_coarse
    
    angle_rad = np.deg2rad(np.clip(roi_angle, -89.9, 89.9))
    slope = np.tan(angle_rad)
    intercept = roi_center_y - slope * (w / 2)
    
    y_pad = padding / (abs(np.cos(angle_rad)) + 1e-6)
    p1 = [0, intercept - y_pad]
    p2 = [w - 1, slope * (w - 1) + intercept - y_pad]
    p3 = [w - 1, slope * (w - 1) + intercept + y_pad]
    p4 = [0, intercept + y_pad]
    
    src_pts = np.float32([p1, p2, p3, p4])
    roi_h = int(2 * padding)
    if roi_h <= 0: return None, None, None, None
    
    dst_pts = np.float32([[0, 0], [w - 1, 0], [w - 1, roi_h - 1], [0, roi_h - 1]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
    roi = cv2.warpPerspective(frame, M, (w, roi_h))
    return roi, M, M_inv, (roi_h, w)

def refine_horizon_stage3(frame, coarse_y, coarse_angle, grad_score=80):
    padding = _calculate_dynamic_padding(grad_score)
    roi_img, M, M_inv, roi_dims = _create_roi(frame, coarse_y, coarse_angle, padding)
    if roi_img is None: return None
    
    refined_local = _run_dual_fusion_pipeline(roi_img)
    if refined_local is None: return None
    
    local_y, local_angle = refined_local
    final_angle = coarse_angle + local_angle
    
    pt_roi = np.array([[[roi_dims[1] / 2, local_y]]], dtype=np.float32)
    pt_global = cv2.perspectiveTransform(pt_roi, M_inv)
    final_y = pt_global[0][0][1]
    
    return final_y, final_angle