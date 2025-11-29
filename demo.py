import torch
import cv2
import numpy as np
import os
import sys
import config
from models.dceunet import DCEUNet
from utils.processing import get_coarse_line_from_mask, refine_horizon_stage3

def run_demo():
    print("--- Efficient Sea-Sky Line Detection ---")
    
    if len(sys.argv) < 2:
        print("Usage: python demo.py <image_path>")
        return

    img_path = sys.argv[1]
    
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    
    model = DCEUNet(input_channels=3, num_classes=1).to(device)
    
    if os.path.exists(config.MODEL_PATH):
        try:
            state_dict = torch.load(config.MODEL_PATH, map_location=device, weights_only=False)
            model.load_state_dict(state_dict)
            print("Model loaded.")
        except:
            try:
                state_dict = torch.load(config.MODEL_PATH, map_location=device)
                model.load_state_dict(state_dict)
                print("Model loaded.")
            except:
                print("Error loading model weights.")
                return
    else:
        print("Model file not found.")
        return
        
    model.eval()

    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return
        
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print("Error reading image.")
        return
        
    h_orig, w_orig = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    input_h, input_w = config.IMAGE_HEIGHT, config.IMAGE_WIDTH
    img_resized = cv2.resize(img_rgb, (input_w, input_h))
    
    img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
    
    print("Processing...")
    
    with torch.no_grad():
        output = model(img_tensor)
        prob_map = torch.sigmoid(output).squeeze().cpu().numpy()
    
    prob_map_orig = cv2.resize(prob_map, (w_orig, h_orig))
    binary_mask = (prob_map_orig > 0.5).astype(np.uint8) * 255
    
    coarse_result = get_coarse_line_from_mask(binary_mask, img_bgr)

    vis = img_bgr.copy()
    
    if coarse_result is not None:
        coarse_y, coarse_angle = coarse_result
        final_y, final_angle = coarse_y, coarse_angle
        
        try:
            fusion_result = refine_horizon_stage3(img_bgr, coarse_y, coarse_angle)
            if fusion_result is not None:
                final_y, final_angle = fusion_result
        except:
            pass

        angle_rad = np.deg2rad(final_angle)
        slope = np.tan(angle_rad)
        intercept = final_y - slope * (w_orig / 2)
        
        y1_calc = intercept
        y2_calc = slope * w_orig + intercept
        
        y1 = int(np.clip(y1_calc, -10000, 10000))
        y2 = int(np.clip(y2_calc, -10000, 10000))
        
        cv2.line(vis, (0, y1), (w_orig, y2), (0, 0, 255), 3)
    else:
        print("Detection failed (showing Mask only).")

    print("Displaying result... (Press any key to exit)")
  
    mask_bgr = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
   
    combined_view = np.hstack((mask_bgr, vis))
 
    display_img = combined_view
    disp_h, disp_w = display_img.shape[:2]
 
    max_display_width = 1600 
    if disp_w > max_display_width:
        scale = max_display_width / disp_w
        display_img = cv2.resize(display_img, (0, 0), fx=scale, fy=scale)
    
    cv2.imshow("Left: Mask | Right: Detection Result", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_demo()