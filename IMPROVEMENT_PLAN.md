# Improvement Plan for Military Supply Chain Optimization System

## 1. Architectural Changes

### Current State
*   **Architecture**: Monolithic Django App + disconnected Jupyter Notebooks.
*   **Model**: Simple CNN training on 32x32 images.
*   **Inference**: Resizes **entire input image** to 32x32, predicts a single class (e.g., "Forest").
*   **Issues**: No route planning, false positives on Face/Random images, low resolution analysis.

### Proposed Architecture

1.  **Service Layer Pattern**: Introduce a `services/` module in Django to handle business logic, separating it from `views.py`.
    *   `ImageValidatorService`: For detecting non-SAR images.
    *   `TerrainAnalysisService`: For patching images and generating cost maps.
    *   `RoutingService`: For A* / Dijkstra pathfinding.
2.  **Patch-Based processing Pipeline**:
    *   Instead of resizing the *whole* image to 32x32, we will:
        *   Keep the image at reasonable resolution (e.g., 512x512).
        *   Slice it into a grid of 32x32 tiles.
        *   Run the CNN on *each* tile.
        *   Reassemble classifications into a **Terrain Matrix**.
3.  **Modern ML Stack**:
    *   Upgrade to TensorFlow 2.x (currently 1.14).
    *   Use `tf.data` for efficient batch processing of patches.

---

## 2. Implementation Steps

### Step 1: Input Validation (rejecting non-SAR)
**Goal**: Prevent processing of selfies/random photos.
**Method**:
*   **Histogram Check**: SAR images have specific intensity distributions (Rayleigh/Gamma). Real photos have wider color variety.
*   **Binary Classifier (Quick Win)**: Train a lightweight SVM or small CNN on (SAR Dataset vs Random Image Dataset) to classify `is_valid_sar`.

### Step 2: Multi-class Terrain & Patch Processing
**Goal**: Generate a map, not a label.
**Algorithm**:
1.  Load Image $I$ ($W \times H$).
2.  Pad $I$ to be multiple of patch size $P$ (32).
3.  Extract patches $P_{i,j}$.
4.  Batch predict: $C_{i,j} = Model(P_{i,j})$.
5.  Map classes to costs:
    *   `Street/Urban` -> Cost: 1 (Preferred)
    *   `Agri/Grassland` -> Cost: 2 (OK)
    *   `Rocky/Mountain` -> Cost: 5 (Hard)
    *   `Flooded/Sea` -> Cost: $\infty$ (Obstacle)

### Step 3: Route Prediction (Path Planning)
**Goal**: Find optimal path from A to B.
**Algorithm (A*)**:
1.  **Nodes**: Each patch center is a node.
2.  **Edges**: Connect to 8 neighbors (N, S, E, W, NE, NW, SE, SW).
3.  **Cost Function**: $F(n) = G(n) + H(n)$
    *   $G(n)$: Real cost to reach node (sum of terrain costs).
    *   $H(n)$: Heuristic (Euclidean distance to target).
4.  **Output**: List of coordinates $(x,y)$ forming the path.

### Step 4: Explainable AI (XAI) Upgrade
**Goal**: Explain *why* a route was chosen (or why an area was avoided).
**Method**:
*   Run LIME *only* on critical patches (e.g., a "Flooded" patch blocking a direct line).
*   Visual output: "Route diverted here because confidence for 'Flooded' is 98%."

---

## 3. Pseudocode

### A. Patch Extraction & Cost Map
```python
def generate_cost_map(image_path, model, patch_size=32):
    full_image = cv2.imread(image_path)
    # resize to ensures grid inputs
    h, w, _ = full_image.shape
    rows = h // patch_size
    cols = w // patch_size
    
    patches = []
    coords = []
    
    for r in range(rows):
        for c in range(cols):
            y1, y2 = r*patch_size, (r+1)*patch_size
            x1, x2 = c*patch_size, (c+1)*patch_size
            patch = full_image[y1:y2, x1:x2]
            patch = cv2.resize(patch, (32, 32)) # Ensure model input size
            patches.append(patch)
            coords.append((r, c))
            
    # Batch Predict
    # Normalization / Preprocessing here
    preds = model.predict(np.array(patches))
    class_indices = np.argmax(preds, axis=1)
    
    # Cost Mapping
    cost_map = np.zeros((rows, cols))
    COSTS = {'street': 1, 'agri': 5, 'sea': 9999, ...}
    
    for (r, c), class_idx in zip(coords, class_indices):
        label = LABELS[class_idx]
        cost_map[r, c] = COSTS[label]
        
    return cost_map
```

### B. Route Planning (A*)
```python
def find_route(cost_map, start_node, end_node):
    # Using 'pathfinding' library or custom A*
    grid = Grid(matrix=cost_map)
    start = grid.node(*start_node)
    end = grid.node(*end_node)
    
    finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
    path, runs = finder.find_path(start, end, grid)
    
    return path # List of (x, y) tuples
```

---

## 4. UI Recommendations
1.  **Split Screen**: Left side = Original Image. Right side = Analyzed Grid/Heatmap.
2.  **Interactive Endpoints**: Allow user to click "Start" and "End" on the map.
3.  **Route Overlay**: Draw the calculated path as a thick polyline on the original image.
4.  **Alerts**: "Warning: Route passes near Flooded area."
