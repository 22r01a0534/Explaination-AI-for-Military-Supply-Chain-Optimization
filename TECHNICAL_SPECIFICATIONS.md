# Technical Specifications: Military Supply Chain Optimization System

## 1. System Requirements

### Hardware Requirements
To run the analysis engine and web server effectively:

*   **Processor (CPU)**: 
    *   *Minimum*: Intel Core i5 (6th Gen) or AMD Ryzen 5 (Ensure AVX instruction support for TensorFlow).
    *   *Recommended*: Intel Core i7 or AMD Ryzen 7 for faster image segmentation and pathfinding.
*   **Memory (RAM)**:
    *   *Minimum*: 8 GB.
    *   *Recommended*: 16 GB (Deep learning models and high-res image arrays consume significant memory).
*   **Storage**:
    *   Approx. 1 GB free space for project files, Python environments, and model weights.
*   **Graphics (GPU) - Optional**:
    *   NVIDIA GPU with CUDA support can significantly speed up the CNN inference, but the system is optimized to run on CPU.

### Software Requirements
*   **Operating System**: Windows 10/11, Linux (Ubuntu 20.04+), or macOS.
*   **Programming Language**: Python 3.7 - 3.10.
*   **Web Framework**: Django (Python).
*   **Key Libraries**:
    *   **TensorFlow / Keras**: For Deep Learning model execution.
    *   **OpenCV (`cv2`)**: For image pre-processing, resizing, and drawing overlays.
    *   **NumPy**: For high-performance matrix operations and grid management.
    *   **Matplotlib**: For generating data visualizations and graphs.
    *   **Scikit-Image**: For advanced image utility functions.
    *   **LIME**: For Explainable AI model interpretation.

---

## 2. Algorithms & Methodologies

The system employs a multi-stage algorithmic pipeline to process satellite imagery and optimize logistics:

### A. Deep Learning: Convolutional Neural Network (CNN)
*   **Purpose**: Automatic Terrain Classification.
*   **Function**: The system scans the uploaded satellite image (SAR/Optical) by breaking it into small patches (e.g., 50x50 pixels).
*   **Architecture**: A custom CNN (likely VGG-style or ResNet-based) trained to recognize topographical features.
*   **Classes**: Classifies regions into 10+ categories: `Urban`, `Forest`, `Water`, `Road`, `Mountain`, `Grassland`, etc.
*   **Output**: A "Class Matrix" and "Confidence Matrix" mapping the terrain type of every grid cell.

### B. Pathfinding: A* (A-Star) Search Algorithm
*   **Purpose**: Optimal Route Calculation.
*   **Function**: Determines the most efficient path from specific Start to End coordinates.
*   **Why A*?**: It guarantees the shortest path (lowest cost) while being faster than Dijkstra's algorithm due to heuristics.
*   **Mechanics**:
    *   **Cost Function ($f(n) = g(n) + h(n)$)**: combining actual travel cost ($g$) and estimated distance to goal ($h$).
    *   **Heuristic**: Euclidean Distance is used to guide the search towards the destination.
    *   **Dynamic Costs**: The cost ($g$) of traversing a cell is weighted by the **Mission Profile**.
        *   e.g., In "Stealth Mode", *Open Roads* have a high cost penalty, while *Forests* have a low cost.
        *   In "Speed Mode", *Roads* have the lowest cost.

### C. Risk Analysis & Logic Heuristics
*   **Purpose**: Mission Safety Assessment.
*   **Function**:
    *   **Risk Mapping**: A lookup table maps terrain types to risk levels (e.g., `Water` = Unsafe, `Urban` = High Risk, `Forest` = Moderate).
    *   **Mission Profiles**: Adjustable weights that modify the `cost_matrix` before pathfinding begins, effectively "re-programming" the A* algorithm's behavior based on user input.

### D. Explainable AI (XAI): LIME
*   **Algorithm**: **L**ocal **I**nterpretable **M**odel-agnostic **E**xplanations.
*   **Purpose**: Trust & Transparency.
*   **Function**:
    1.  Identifies a "Critical Patch" (e.g., a high-risk area the path crosses).
    2.  Perturbs (randomly modifies) the image patch thousands of times.
    3.  Observes how the CNN's prediction changes.
    4.  Generates a heatmap showing exactly *which pixels* (e.g., the edge of a building or a body of water) caused the model to classify it as such.
