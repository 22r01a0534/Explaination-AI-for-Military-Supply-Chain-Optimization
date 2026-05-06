# Project Report - Part 2

## 11. Testing + Results

### Testing Approach
To ensure the reliability of the system, a multi-layered testing strategy was employed:
1.  **Unit Testing**:
    *   **Algorithm Validation**: The A* Pathfinding algorithm was tested on abstract grid maps with known obstacles to verify it always calculates the shortest valid path.
    *   **Model Input/Output**: The CNN loader was tested to ensure it correctly normalizes images and outputs probability vectors of the expected shape.
2.  **Integration Testing**:
    *   Verified the data flow between the Django Frontend (View), the Logic Module, and the TensorFlow backend. Confirmed that user-selected coordinates (pixels) map correctly to the internal grid cells.
3.  **System Testing**:
    *   End-to-end testing involved uploading various satellite images (desert, urban, forest) and simulating different mission profiles to observe changes in the generated path.

### Results
*   **Model Accuracy**: The Convolutional Neural Network (CNN) achieved an accuracy of approximately **88%** on the test dataset, effectively distinguishing between critical classes like `Water` (Obstacle) and `Road` (Safe).
*   **Pathfinding Efficiency**: The system successfully generates optimal routes in under **5 seconds** for standard 300x300 resolutions. It correctly demonstrates "Risk Avoidance" by routing around hazardous zones defined in the Mission Profile.
*   **Visual Output**: The system produces clear, color-coded visual overlays where the path color indicates risk levels (Green: Safe, Orange: Moderate, Red: High Risk).
*   **Object Detection**: The system successfully lists all terrain types encountered along the route and within the global scan area, providing comprehensive situational awareness.

## 12. Challenges Faced + Solutions

during the development of the project, several technical challenges were encountered and resolved:

### Challenge 1: High Computational Cost of Image Processing
*   **Issue**: Processing full-resolution satellite imagery (e.g., 4K resolution) was extremely slow and led to memory overflow errors, making the application unresponsive.
*   **Solution**: We implemented a pre-processing pipeline that resizes images to a standardized **300x300 pixel** resolution. This creates a manageable grid size for the A* algorithm and CNN patch extraction without significantly compromising tactical utility.

### Challenge 2: Threading Conflicts with Visualization Libraries
*   **Issue**: The integration of `Matplotlib` for generating LIME explanations caused the Django server to crash with a `RuntimeError: main thread is not in main loop`. This occurs because Matplotlib defaults to an interactive GUI backend which is not thread-safe for web servers.
*   **Solution**: We explicitly configured Matplotlib to use the **'Agg' backend** (Anti-Grain Geometry). This non-interactive backend renders images directly to memory buffers/files, bypassing the GUI event loop and resolving the crash.

### Challenge 3: Distinguishing Similar Terrains
*   **Issue**: The CNN initially struggled to differentiate between visually similar terrains, such as `Grassland` and `Agri` (Agricultural Land), leading to suboptimal cost assignment.
*   **Solution**: We improved the model by applying **Data Augmentation** techniques (rotation, flipping, and contrast adjustment) during training and adjusted the cost weights so that minor misclassifications between low-risk terrains had minimal impact on the final path.

## 13. Conclusion + Future Scope

### Conclusion
The **AI-Driven Military Supply Chain Optimization System** successfully meets its primary objectives. It demonstrates that Deep Learning can effectively automate the tedious task of map analysis, while A* search algorithms can mathematically optimize logistics based on complex mission constraints. The addition of Explainable AI (LIME) and detailed risk reporting transforms the system from a simple calculator into a trustworthy decision-support tool for commanders. By reducing the time required for route planning and increasing safety through objective risk assessment, this system has the potential to significantly enhance operational efficiency.

### Future Scope
While the current system is functional, there are several avenues for future enhancement:
1.  **Real-Time Drone Feed Integration**: Adapting the system to process live video feeds rather than static image uploads, allowing for dynamic re-routing if a new threat (e.g., an explosion or roadblock) is detected in real-time.
2.  **3D Elevation Analysis**: Integrating Digital Elevation Models (DEM) to account for terrain slope. This is critical for heavy vehicles that cannot climb steep gradients, even if the terrain type (e.g., Grassland) is theoretically passable.
3.  **Multi-Agent Coordination**: upgrading the pathfinding engine to handle multiple convoys simultaneously, ensuring they do not cross paths at bottlenecks or cause congestion.
4.  **Weather Integration**: Pulling real-time weather data to dynamically adjust terrain costs (e.g., `Dirt Road` becomes `Mud` (High Cost) during rain).
