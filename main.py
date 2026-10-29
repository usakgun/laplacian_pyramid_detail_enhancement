import cv2
import numpy as np
import os
import glob

def resize_min_edge(img, min_edge=512):
    h, w = img.shape[:2]
    scale = min_edge / min(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

def build_pyramids(img, levels=5):
    G = img.astype(np.float32)
    gaussian_pyr = [G]
    
    for i in range(levels - 1):
        G = cv2.pyrDown(G)
        gaussian_pyr.append(G)
        
    laplacian_pyr = []
    
    for i in range(levels - 1):
        target_size = (gaussian_pyr[i].shape[1], gaussian_pyr[i].shape[0])
        GE = cv2.pyrUp(gaussian_pyr[i+1], dstsize=target_size)
        L = cv2.subtract(gaussian_pyr[i], GE)
        laplacian_pyr.append(L)
        
    laplacian_pyr.append(gaussian_pyr[-1])
    return laplacian_pyr

def enhance_and_reconstruct(laplacian_pyr, weights):
    enhanced_pyr = []
    
    for i in range(len(laplacian_pyr) - 1):
        enhanced_pyr.append(laplacian_pyr[i] * weights[i])
    enhanced_pyr.append(laplacian_pyr[-1] * weights[-1])
    
    img_recon = enhanced_pyr[-1]
    
    for i in range(len(enhanced_pyr) - 2, -1, -1):
        target_size = (enhanced_pyr[i].shape[1], enhanced_pyr[i].shape[0])
        img_recon = cv2.pyrUp(img_recon, dstsize=target_size)
        img_recon = cv2.add(img_recon, enhanced_pyr[i])
        
    return np.clip(img_recon, 0, 255).astype(np.uint8)

def process_dataset(input_dir="images", output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    extensions = ('*.jpg', '*.jpeg', '*.png')
    image_paths = []
    
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        
    beta_weights = [1.0, 1.05, 1.15, 1.20, 1.0]
    
    for path in image_paths:
        filename = os.path.basename(path)
        img = cv2.imread(path)
        
        if img is None:
            continue
            
        print(f"Processing: {filename}")
        
        resized_img = resize_min_edge(img)
        laplacian_pyr = build_pyramids(resized_img)
        enhanced_img = enhance_and_reconstruct(laplacian_pyr, beta_weights)
        
        name, ext = os.path.splitext(filename)
        out_path = os.path.join(output_dir, f"{name}_enhanced{ext}")
        cv2.imwrite(out_path, enhanced_img)

if __name__ == "__main__":
    process_dataset()