# System Diagrams

## 1. Architecture Diagram
This diagram outlines the high-level structure of the Military Supply Chain Optimization System, separating the Frontend, Backend, AI Engine, and Data Storage.

```mermaid
graph TD
    subgraph Client_Side
        Browser[Web Browser]
        UI[User InterfaceHTML/JS/CSS]
    end

    subgraph Server_Side
        Django[Django Web Server]
        Views[View Controllers]
        Models[Data Models]
    end

    subgraph AI_Engine
        Preprocess[Image Preprocessing (OpenCV)]
        CNN[Terrain Classifier (TensorFlow/Keras)]
        Pathfinding[A* Path Optimization]
        XAI[LIME Explainer]
    end

    subgraph Data_Layer
        DB[(SQLite/PostgreSQL)]
        FileSystem[File Storage (Images/Models)]
    end

    Browser -->|HTTP Request| Django
    Django --> Views
    Views -->|Query/Save| Models
    Models -->|Read/Write| DB
    Views -->|Image Data| Preprocess
    Preprocess --> CNN
    CNN --> Pathfinding
    Pathfinding --> XAI
    Views -->|Save Results| FileSystem
    XAI -->|Explanation Data| Views
    Pathfinding -->|Route Data| Views
    Views -->|Response| Browser
```

## 2. Class Diagram
This diagram details the internal structure of the system, including the Django models for data persistence, the View functions for handling requests, and the Logic module for AI processing.

```mermaid
classDiagram
    class Mission {
        +int id
        +User user
        +DateTime date_created
        +String mission_type
        +String status
        +float risk_score
        +String image_path
        +String ranked_personnel
        +__str__()
    }

    class Views {
        +index(request)
        +history_view(request)
        +delete_mission(request, id)
        +Optimize(request)
        +OptimizeAction(request)
        +PerformAnalysis(request)
    }

    class Logic_AI_Engine {
        -_COSTS: Dict
        -_MISSION_PROFILES: Dict
        -_RISK_MAPPING: Dict
        +load_prediction_model(path)
        +validate_image_is_sar(image)
        +process_image_patches(model, img, mission_type)
        +get_dynamic_costs(mission_type)
        +astar_path(cost_matrix, start, end)
        +analyze_path_risk(path, class_matrix)
        +explain_critical_patch(model, img, path)
        +visualize_results(img, matrix, path)
        +heuristic(a, b)
    }

    Views --> Mission : CRUD Operations
    Views --> Logic_AI_Engine : Calls AI Functions
    Logic_AI_Engine ..> ExternalLibs : Uses
    class ExternalLibs {
        <<TensorFlow>>
        <<OpenCV>>
        <<NumPy>>
        <<LIME>>
    }
```

## 3. Use Case Diagram
This diagram illustrates the interactions between the User (Commander/Logistics Officer) and the system's various functionalities.

```mermaid
usecaseDiagram
    actor Commander as "Mission Commander"

    package "Military Supply Chain System" {
        usecase "Upload Satellite Map" as UC1
        usecase "Validate Image" as UC2
        usecase "Select Mission Profile" as UC3
        usecase "Define Start/End Points" as UC4
        usecase "View Optimized Route" as UC5
        usecase "View Risk Analysis" as UC6
        usecase "View AI Explanation" as UC7
        usecase "Manage Mission History" as UC8
    }

    Commander --> UC1
    UC1 ..> UC2 : <<include>>
    Commander --> UC3
    Commander --> UC4
    Commander --> UC5
    UC5 ..> UC6 : <<include>>
    UC5 ..> UC7 : <<extend>>
    Commander --> UC8
```

## 4. Activity Diagram
This diagram visualizes the flow of control from the moment a user uploads an image to the generation of the final optimized route.

```mermaid
flowchart TD
    Start((Start)) --> Upload[User Uploads Map Image]
    Upload --> Validate{Valid Image?}
    
    Validate -- No --> Error[Show Error Message] --> End((End))
    
    Validate -- Yes --> Preprocess[Resize & Preprocess Image]
    Preprocess --> Params[User Selects Mission Type & Personnel]
    Params --> Points[User Clicks Start & End Points]
    
    Points --> Segment[Divide Image into Patches]
    Segment --> Classify[CNN Classifies Terrain Types]
    
    Classify --> CostMap[Generate Cost Matrix based on Mission]
    CostMap --> AStar[Execute A* Pathfinding]
    
    AStar --> RouteFound{Route Found?}
    
    RouteFound -- No --> Fail[Report Obstruction] --> SaveDB
    
    RouteFound -- Yes --> Risk[Calculate Risk Metrics]
    Risk --> Explain[Generate LIME Explanation]
    Explain --> Visualize[Overlay Path on Image]
    Visualize --> Display[Display Results to User]
    
    Display --> SaveDB[Save Mission to Database]
    SaveDB --> End
```

## 5. Entity-Relationship (ER) Diagram
This diagram represents the database schema and the relationships between the data entities in the system.

```mermaid
erDiagram
    USER ||--o{ MISSION : "initiates"
    
    USER {
        int id PK
        string username
        string password
        string email
        string first_name
        string last_name
        boolean is_staff
        boolean is_active
        datetime date_joined
    }

    MISSION {
        int id PK
        int user_id FK "Nullable"
        datetime date_created
        string mission_type
        string status
        float risk_score
        string image_path
        string ranked_personnel
    }
```
