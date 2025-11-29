import torch
import cv2
import numpy as np
import os
import argparse
import time
import config
from models.dceunet import DCEUNet
from utils.processing import get_coarse_line_from_mask, refine_horizon_stage3

def run_video_demo(video_path):
    print(f"--- Efficient Sea-Sky Line Detection ---")
    
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = DCEUNet(input_channels=3, num_classes=1).to(device)
    
    if os.path.exists(config.MODEL_PATH):
        try:
            state_dict = torch.load(config.MODEL_PATH, map_location=device, weights_only=False)
            model.load_state_dict(state_dict)
            print("Model loaded.")
        except:
            state_dict = torch.load(config.MODEL_PATH, map_location=device)
            model.load_state_dict(state_dict)
    else:
        print("Error: Weights not found.")
        return
    
    model.eval()

    use_fp16 = (device.type == 'cuda')
    if use_fp16:
        print("FP16 Enabled")
        model.half()
    else:
        print("FP16 not supported on CPU. Using standard FP32.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Source: {width}x{height} @ {fps:.1f} FPS")
    print("Processing mode: Post-Processing")

    prev_time = time.time()
    fps_buffer = []
    
    last_y, last_angle = None, None

    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        
        input_h, input_w = config.IMAGE_HEIGHT, config.IMAGE_WIDTH
        img_resized = cv2.resize(frame, (input_w, input_h))
        
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
        
        if use_fp16:
            img_tensor = img_tensor.half() / 255.0
        else:
            img_tensor = img_tensor.float() / 255.0
        
        with torch.no_grad():
            output = model(img_tensor)
            prob_map = torch.sigmoid(output).squeeze().float().cpu().numpy()
        
        prob_map_orig = cv2.resize(prob_map, (width, height))
        binary_mask = (prob_map_orig > 0.5).astype(np.uint8) * 255
        
        coarse_result = get_coarse_line_from_mask(binary_mask, frame)
        
        final_y, final_angle = None, None
        
        if coarse_result is not None:
            coarse_y, coarse_angle = coarse_result
            final_y, final_angle = coarse_y, coarse_angle
            
            try:
                fusion_result = refine_horizon_stage3(frame, coarse_y, coarse_angle)
                if fusion_result is not None:
                    final_y, final_angle = fusion_result
            except:
                pass
        
        if final_y is not None:
            if last_y is not None:
                final_y = 0.7 * final_y + 0.3 * last_y
                final_angle = 0.7 * final_angle + 0.3 * last_angle
            
            last_y, last_angle = final_y, final_angle
            
            angle_rad = np.deg2rad(final_angle)
            slope = np.tan(angle_rad)
            intercept = final_y - slope * (width / 2)
            
            y1_calc = intercept
            y2_calc = slope * width + intercept
 
            y1 = int(np.clip(y1_calc, -10000, 10000))
            y2 = int(np.clip(y2_calc, -10000, 10000))
            
            cv2.line(frame, (0, y1), (width, y2), (0, 0, 255), 3)
            
        elif last_y is not None:
            angle_rad = np.deg2rad(last_angle)
            slope = np.tan(angle_rad)
            intercept = last_y - slope * (width / 2)
            
            y1_calc = intercept
            y2_calc = slope * width + intercept
            
            y1 = int(np.clip(y1_calc, -10000, 10000))
            y2 = int(np.clip(y2_calc, -10000, 10000))
            
            cv2.line(frame, (0, y1), (width, y2), (0, 0, 255), 3)

        curr_time = time.time()
        process_time = curr_time - prev_time
        prev_time = curr_time
        
        current_fps = 1.0 / (process_time + 1e-6)
        fps_buffer.append(current_fps)
        if len(fps_buffer) > 10: fps_buffer.pop(0)
        avg_fps = sum(fps_buffer) / len(fps_buffer)
        
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        display_frame = frame
        if width > 1280:
            scale = 1280 / width
            display_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            
        cv2.imshow('Detection', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to input video file')
    args = parser.parse_args()
    
    if os.path.exists(args.input):
        run_video_demo(args.input)
    else:
        print(f"Input video not found: {args.input}")