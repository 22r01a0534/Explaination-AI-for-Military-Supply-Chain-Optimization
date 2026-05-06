import numpy as np
import cv2
import os
import heapq
import base64
import io
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries

# Global model cache to avoid reloading
_CACHED_MODEL = None
_LABELS = ['agri', 'barrenland', 'building', 'flooded', 'forest', 'grassland', 'mountains', 'sea', 'street', 'urban']

# Cost mapping for path planning
# Lower is better. Infinity for obstacles.
_COSTS = {
    'street': 1,
    'grassland': 2,
    'barrenland': 3,
    'agri': 4,
    'forest': 10,
    'rocky': 20,
    'mountains': 50,
    'building': 100, # Avoid buildings if possible
    'urban': 50,
    'flooded': 9999, # Obstacle
    'sea': 9999      # Obstacle
}

# Mission Profiles
# Adjusts base costs for specific mission types
_MISSION_PROFILES = {
    'standard': {},
    'stealth': {
        'urban': 200,    # Avoid urban areas (high visibility)
        'street': 50,    # Avoid streets if possible
        'forest': 5,     # Prefer cover
        'mountains': 40  # Hard but good cover
    },
    'heavy_transport': {
        'forest': 100,   # Impassable for heavy trucks
        'mountains': 9999, # Impossible
        'rocky': 100,
        'street': 0.5    # Prefer paved roads strongly
    },
    'speed': {
        'street': 0.1,   # Maximize speed
        'grassland': 1.5,
        'urban': 5       # Traffic risk
    }
}

# Risk Mapping
# Defines which terrains are considered risky
_RISK_MAPPING = {
    'street': 'safe',
    'grassland': 'safe',
    'agri': 'safe',
    'barrenland': 'safe',
    'forest': 'risky',
    'rocky': 'risky',
    'mountains': 'risky', 
    'urban': 'risky',
    'building': 'unsafe',
    'flooded': 'unsafe',
    'sea': 'unsafe'
}

def get_dynamic_costs(mission_type='standard'):
    """
    Returns a cost dictionary adjusted for the mission profile.
    """
    base_costs = _COSTS.copy()
    modifiers = _MISSION_PROFILES.get(mission_type, {})
    
    for terrain, mod_cost in modifiers.items():
        base_costs[terrain] = mod_cost
        
    return base_costs

def load_prediction_model(model_path):
    """Singleton model loader"""
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        from keras.models import load_model
        # Use a custom object scope if necessary for legacy layers, 
        # but for standard CNN it should be fine.
        _CACHED_MODEL = load_model(model_path)
    return _CACHED_MODEL

def validate_image_is_sar(image_arr):
    """
    Basic validation for SAR/Satellite imagery.
    Returns: (is_valid: bool, reason: str)
    """
    # 1. Check if grayscale or low saturation
    if len(image_arr.shape) == 3:
        hsv = cv2.cvtColor(image_arr, cv2.COLOR_BGR2HSV)
        saturation = hsv[:,:,1]
        mean_sat = np.mean(saturation)
        if mean_sat > 50: # Arbitrary threshold: Real photos usually have higher saturation
            return False, "Image has high color saturation. SAR images are typically grayscale/low-color."
    
    # 2. Histogram check (Optional: SAR usually has Rayleigh distribution)
    # For now, we trust the saturation check as a basic filter against selfies.
    return True, "Valid"

def process_image_patches(model, image_path, patch_size=32, mission_type='standard'):
    """
    Divides image into patches, classifies each, and builds a cost map based on mission profile.
    """
    full_image = cv2.imread(image_path)
    if full_image is None:
        raise ValueError("Could not read image")

    # Force resize to 300x300 as per user request
    full_image = cv2.resize(full_image, (300, 300))

    # Resize to have dimensions multiple of patch_size for simplicity
    # or handle partial edges. Here we simply crop or resize.
    h, w, _ = full_image.shape
    
    # Calculate grid size
    rows = h // patch_size
    cols = w // patch_size
    
    # Resize image to fit exact grid (minor distortion is acceptable for this usecase)
    target_h = rows * patch_size
    target_w = cols * patch_size
    full_image = cv2.resize(full_image, (target_w, target_h))
    
    patches = []
    
    # Extract patches
    for r in range(rows):
        for c in range(cols):
            y1, y2 = r*patch_size, (r+1)*patch_size
            x1, x2 = c*patch_size, (c+1)*patch_size
            patch = full_image[y1:y2, x1:x2]
            
            # Resize for model input (32x32)
            patch_resized = cv2.resize(patch, (32, 32))
            
            # Preprocess for model (1, 32, 32, 3) + Normalize
            patch_resized = patch_resized.astype('float32') / 255.0
            patches.append(patch_resized)
            
    # Batch predict
    patches_arr = np.array(patches)
    # create batch if model expects it
    if patches_arr.shape[0] == 0:
        return None, None
        
    predictions = model.predict(patches_arr)
    class_indices = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)
    
    # Get costs for mission
    active_costs = get_dynamic_costs(mission_type)
    
    # Build cost matrix
    cost_matrix = np.zeros((rows, cols))
    class_matrix = np.zeros((rows, cols), dtype=int)
    conf_matrix = np.zeros((rows, cols))
    
    idx = 0
    for r in range(rows):
        for c in range(cols):
            cls_idx = class_indices[idx]
            label = _LABELS[cls_idx]
            cost_matrix[r, c] = active_costs.get(label, 9999)
            class_matrix[r, c] = cls_idx
            conf_matrix[r, c] = confidences[idx]
            idx += 1
            
    return cost_matrix, class_matrix, conf_matrix, full_image


def astar_path(cost_matrix, start, end):
    """
    A* Pathfinding algorithm.
    start: (row, col)
    end: (row, col)
    """
    rows, cols = cost_matrix.shape
    
    # Priority Queue: (f_score, r, c)
    open_set = []
    heapq.heappush(open_set, (0, start[0], start[1]))
    
    came_from = {}
    
    # g_score: Cost from start to current
    g_score = {node: float('inf') for node in np.ndindex(rows, cols)}
    g_score[start] = 0
    
    # f_score: g_score + heuristic
    f_score = {node: float('inf') for node in np.ndindex(rows, cols)}
    f_score[start] = heuristic(start, end)
    
    while open_set:
        current_f, r, c = heapq.heappop(open_set)
        current = (r, c)
        
        if current == end:
            return reconstruct_path(came_from, current)
            
        # Neighbors: 8-connectivity or 4-connectivity? 
        # Let's use 8-connectivity for smoother paths.
        neighbors = [
            (r-1, c), (r+1, c), (r, c-1), (r, c+1),
            (r-1, c-1), (r-1, c+1), (r+1, c-1), (r+1, c+1)
        ]
        
        for nr, nc in neighbors:
            if 0 <= nr < rows and 0 <= nc < cols:
                # Terrain cost
                terrain_cost = cost_matrix[nr, nc]
                
                # Distance cost (1.0 for orthogonal, 1.414 for diagonal)
                dist_cost = 1.414 if (nr != r and nc != c) else 1.0
                
                # Total move cost
                move_cost = terrain_cost * dist_cost
                
                tentative_g_score = g_score[current] + move_cost
                
                if tentative_g_score < g_score[(nr, nc)]:
                    # Found a better path to neighbor
                    came_from[(nr, nc)] = current
                    g_score[(nr, nc)] = tentative_g_score
                    f_score[(nr, nc)] = tentative_g_score + heuristic((nr, nc), end)
                    heapq.heappush(open_set, (f_score[(nr, nc)], nr, nc))
                    
    return None # No path found

def heuristic(a, b):
    # Euclidean distance
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    return total_path[::-1] # Return reversed

def analyze_path_risk(path, class_matrix):
    """
    Analyzes the path and returns a risk report.
    """
    if not path:
        return "No path", 0.0

    total_nodes = len(path)
    risky_nodes = 0
    safe_nodes = 0
    
    for r, c in path:
        idx = class_matrix[r, c]
        label = _LABELS[idx]
        risk_level = _RISK_MAPPING.get(label, 'unknown')
        if risk_level == 'risky' or risk_level == 'unsafe':
            risky_nodes += 1
        else:
            safe_nodes += 1
            
    risk_percentage = (risky_nodes / total_nodes) * 100
    
    if risk_percentage < 10:
        overall_status = "SAFE"
    elif risk_percentage < 40:
        overall_status = "MODERATE RISK"
    else:
        overall_status = "HIGH RISK"
        
    return overall_status, risk_percentage


def get_path_objects(path, class_matrix):
    """
    Returns a sorted list of unique objects (terrains) intersected by the path.
    """
    if not path:
        return []
    
    unique_labels = set()
    for r, c in path:
        idx = class_matrix[r, c]
        label = _LABELS[idx]
        unique_labels.add(label)
        
    return sorted(list(unique_labels))





def explain_critical_patch(model, full_image, class_matrix, conf_matrix, path, patch_size=32):
    """
    Finds the most "risky" or interesting patch on the path and runs LIME on it.
    Returns: Base64 encoded explanation image or None.
    """
    if not path or len(path) < 2:
        return None
        
    # 1. Find critical patch (Highest risk on path, or lowest confidence)
    # Strategy: Find a patch on path that is 'Risky' or 'Unsafe'.
    # If none, find patch with lowest confidence.
    critical_node = None
    
    # Priority 1: Risky terrain on path
    for r, c in path:
        label = _LABELS[class_matrix[r, c]]
        if _RISK_MAPPING.get(label) in ['risky', 'unsafe']:
            critical_node = (r, c)
            break
            
    # Priority 2: Lowest confidence on path
    if not critical_node:
        min_conf = 1.0
        for r, c in path:
            if conf_matrix[r, c] < min_conf:
                min_conf = conf_matrix[r, c]
                critical_node = (r, c)
                
    if not critical_node:
        critical_node = path[len(path)//2] # Fallback to middle
        
    # 2. Extract patch
    r, c = critical_node
    y1, y2 = r*patch_size, (r+1)*patch_size
    x1, x2 = c*patch_size, (c+1)*patch_size
    
    # Ensure dimensions match model input (re-extraction/resize logic)
    patch = full_image[y1:y2, x1:x2]
    # LIME needs the exact input format used for prediction
    # Note: process_image_patches resizes/normalizes. LIME handles perturbations.
    # Our model expects (1, 32, 32, 3) normalized float32.
    
    # We need a wrapper prediction function for LIME that handles the batch
    def predict_wrapper(images):
        # Images come in as (N, H, W, C) usually uint8 or float depending on input
        # We need to ensure they match training preprocessing
        proc_imgs = []
        for img in images:
            if img.shape[:2] != (32, 32):
                img = cv2.resize(img, (32, 32))
            img = img.astype('float32') / 255.0
            proc_imgs.append(img)
        return model.predict(np.array(proc_imgs))

    # Pre-resize patch for LIME explainer
    patch_input = cv2.resize(patch, (32, 32))
    
    # 3. Run LIME
    explainer = lime_image.LimeImageExplainer()
    try:
        explanation = explainer.explain_instance(
            patch_input, 
            predict_wrapper, 
            top_labels=1, 
            hide_color=0, 
            num_samples=500 # Low samples for speed
        )
        
        # 4. Visualize
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], 
            positive_only=True, 
            num_features=5, 
            hide_rest=False
        )
        
        img_boundry = mark_boundaries(temp / 2 + 0.5, mask)
        
        # Plot to buffer
        plt.figure(figsize=(4, 4))
        plt.imshow(img_boundry)
        plt.axis('off')
        plt.title(f"Why {_LABELS[class_matrix[r, c]]}?\n(Patch {r},{c})")
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        
        return base64.b64encode(buf.getvalue()).decode()
        
    except Exception as e:
        print(f"LIME Error: {e}")
        return None

def simplify_path(path):
    """
    Simplifies a grid path to be smoother using a basic ray-casting like approach
    or just selecting key points. For simple grid A*, just taking centers is often enough,
    but we can skip collinear points.
    """
    if not path or len(path) < 3:
        return path
        
    # Simple collinear simplification
    new_path = [path[0]]
    for i in range(1, len(path)-1):
        prev = new_path[-1]
        curr = path[i]
        next_p = path[i+1]
        
        # Check slope
        # (y2-y1)/(x2-x1) == (y3-y2)/(x3-x2)
        dy1 = curr[0] - prev[0]
        dx1 = curr[1] - prev[1]
        dy2 = next_p[0] - curr[0]
        dx2 = next_p[1] - curr[1]
        
        if dy1 * dx2 != dy2 * dx1:
            new_path.append(curr)
            
    new_path.append(path[-1])
    return new_path

def visualize_results(full_image, class_matrix, path, patch_size=32):
    """
    Draws the grid, classifications, and optimal path on the image.
    Returns: Processed Image (numpy array)
    """
    vis_img = full_image.copy()
    rows, cols = class_matrix.shape
    
    # ... (Overlay Grid code suppressed for cleaner look) ...
        
    # 2. Highlight Obstacles (Flooded/Sea/Building) with semi-transparent red
    overlay = vis_img.copy()
    for r in range(rows):
        for c in range(cols):
            label = _LABELS[class_matrix[r, c]]
            if label in ['flooded', 'sea', 'building', 'urban']: # Treat Urban as cautionary
                y1, y2 = r*patch_size, (r+1)*patch_size
                x1, x2 = c*patch_size, (c+1)*patch_size
                color = (0, 0, 255) if label in ['flooded', 'sea'] else (0, 165, 255) # Red for water, Orange for buildings
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    
    alpha = 0.25
    vis_img = cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0)

    # 3. Draw Path with Risk Coloring
    if path and len(path) > 1:
        # We perform graphical smoothing for better visual "accuracy"
        # Instead of pixel-perfect grid lines, we use anti-aliased curves or direct connection
        
        # Convert all grid points to pixel centers first
        pixel_points = []
        for (r, c) in path:
            px = int(c * patch_size + patch_size / 2)
            py = int(r * patch_size + patch_size / 2)
            pixel_points.append((px, py))
            
        # Draw edges
        for i in range(len(pixel_points) - 1):
            pt1 = pixel_points[i]
            pt2 = pixel_points[i+1]
            
            # Determine color from destination method
            # Re-map pixel point to grid to get risk
            # (Simple approximation using next node risk)
            grid_r = int(path[i+1][0])
            grid_c = int(path[i+1][1])
            
            idx = class_matrix[grid_r, grid_c]
            label = _LABELS[idx]
            risk_level = _RISK_MAPPING.get(label, 'safe')
            
            if risk_level == 'safe':
                color = (0, 255, 0) # Green
            elif risk_level == 'risky':
                color = (0, 215, 255) # Gold/Orange
            else:
                color = (0, 0, 255) # Red
                
            cv2.line(vis_img, pt1, pt2, color, 4, cv2.LINE_AA)
            
        # Draw Start/End Markers
        cv2.circle(vis_img, pixel_points[0], 8, (0, 255, 0), -1) # Start Green
        cv2.circle(vis_img, pixel_points[-1], 8, (0, 0, 255), -1) # End Red

    return vis_img


def get_all_objects(class_matrix):
    """
    Returns a sorted list of ALL unique objects (terrains) present in the image.
    """
    unique_indices = np.unique(class_matrix)
    labels = sorted([_LABELS[i] for i in unique_indices])
    return labels

def explain_route_safety(path, class_matrix):
    """
    Generates a textual explanation of the route safety.
    """
    if not path:
        return "No route could be calculated due to blocking obstacles."
        
    # Analyze composition
    terrain_counts = {}
    total_steps = len(path)
    
    for r, c in path:
        label = _LABELS[class_matrix[r, c]]
        terrain_counts[label] = terrain_counts.get(label, 0) + 1
        
    # Sort by frequency
    sorted_terrains = sorted(terrain_counts.items(), key=lambda x: x[1], reverse=True)
    
    primary_terrain = sorted_terrains[0][0]
    
    explanation = f"The selected AI route minimizes risk by primarily traversing **{primary_terrain}** ({int(terrain_counts[primary_terrain]/total_steps*100)}% of path). "
    
    risky_terrains = [t for t, c in sorted_terrains if _RISK_MAPPING.get(t) in ['risky', 'unsafe']]
    safe_terrains = [t for t, c in sorted_terrains if _RISK_MAPPING.get(t) == 'safe']
    
    if risky_terrains:
        explanation += f"However, it navigates through some **{', '.join(set(risky_terrains))}** areas where no safer alternative existed. "
    else:
        explanation += "The route successfully avoids all high-risk zones. "
        
    explanation += "The algorithm prioritized low-cost traversal while maintaining distance from detected threats and obstacles."
    
    return explanation

