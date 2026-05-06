
import numpy as np
import cv2
import os

def generate_samples():
    # Define output directory
    output_dir = "SupplyApp/static/sample_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Common settings
    width, height = 300, 300

    # 1. Urban / Street (geometric patterns)
    img_urban = np.zeros((height, width), dtype=np.uint8) + 50 # Dark background
    
    # Add "streets"
    cv2.line(img_urban, (0, 150), (300, 150), (100), 20)
    cv2.line(img_urban, (150, 0), (150, 300), (100), 20)
    
    # Add "buildings" (bright rectangles)
    for _ in range(20):
        x, y = np.random.randint(0, 300, 2)
        w_b, h_b = np.random.randint(20, 50, 2)
        color = np.random.randint(150, 255)
        cv2.rectangle(img_urban, (x, y), (x+w_b, y+h_b), (int(color)), -1)
        
    cv2.imwrite(os.path.join(output_dir, "sample_urban_sar.png"), img_urban)
    print("Generated sample_urban_sar.png")

    # 2. Forest (Noisy texture)
    img_forest = np.random.normal(60, 15, (height, width)).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "sample_forest_sar.png"), img_forest)
    print("Generated sample_forest_sar.png")

    # 3. Mountains (High contrast ridges - simulated with blurred noise)
    noise = np.random.randint(0, 255, (height, width)).astype(np.uint8)
    img_mountains = cv2.GaussianBlur(noise, (15, 15), 0)
    # Increase contrast
    img_mountains = cv2.normalize(img_mountains, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(os.path.join(output_dir, "sample_mountains_sar.png"), img_mountains)
    print("Generated sample_mountains_sar.png")

    print(f"\nSuccess! 3 Sample images saved to {os.path.abspath(output_dir)}")
    print("These images are Grayscale (SAR-like) and meet the system requirements.")

if __name__ == "__main__":
    generate_samples()
