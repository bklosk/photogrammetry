#  Photogrammetree 🌳 
## Climate Disaster Mitigation 🌍

This project uses **LiDAR** data 🛰️ and **orthophotos** 📸 combined with **3D semantic segmentation** 🤖 and FEA/FDS to identify and analyze mutable candidates for climate risk mitigation.

## The Goal 🎯

Our aim is to pinpoint specific areas or objects within a 3D environment (e.g., vegetation, buildings, infrastructure) that can be modified or managed to reduce climate-related risks like flooding, heat islands, or wildfire spread.

For example, we could:
*   Identify vegetation likely to be a vector for fire spread to buildings.
*   Find optimal placements for floodwalls, sandbags, or home risers.
*   Detect rooftops suitable for conversion to steel roofs 🌿 or solar panel installations ☀️.
*   Assess infrastructure vulnerabilities that need reinforcement.

## Technologies 🛠️

*   **LiDAR:** For creating accurate 3D point clouds of the environment.
*   **Orthophotos:** Georeferenced aerial imagery providing color and texture information.
*   **3D Semantic Segmentation:** Machine learning models to classify objects within the 3D point cloud (e.g., ground, buildings, low vegetation, high vegetation).

## Workflow Overview (Conceptual) 🗺️

1.  **Data Acquisition & Preprocessing:**
    *   Collect LiDAR scans and orthophotos.
    *   Align and fuse these datasets.
    *   Colorize the LiDAR point cloud using orthophoto imagery.
2.  **3D Semantic Segmentation:**
    *   Train or use a pre-trained model to classify points in the cloud (e.g., using Open3D-ML).
3.  **Candidate Identification & Analysis:**
    *   Develop algorithms to analyze the segmented point cloud.
    *   Identify "mutable" objects/areas based on their class and properties.
    *   Evaluate their potential for climate risk mitigation.
4.  **Physics-Based Simulation & ML Analysis 🌪️🔥🌊:**
    *   Run Finite Element Analysis (FEA) and Fire Dynamics Simulator (FDS) simulations on the segmented point cloud.
    *   Simulate physical phenomena like fires, floods, and high winds.
    *   Utilize Machine Learning to analyze simulation results and refine mitigation strategies.
5.  **Visualization & Reporting:**
    *   Visualize the results in 3D.
    *   Generate reports for decision-making.

---

## Project Setup and Execution

### Prerequisites

- Python 3.10 or higher
- Git (for cloning the repository, if applicable)

### Setup in a GitHub Codespace or Local Environment

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone <repository-url>
    cd photogrammetry
    ```

2.  **Create and activate a Python virtual environment:**
    It's highly recommended to use a virtual environment to manage project dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
    (On Windows, use `venv\Scripts\activate`)

3.  **Install Python dependencies:**
    The required Python packages are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

### Running the Pipeline

The main pipeline script is `scripts/pipeline.py`. This script processes a hardcoded point cloud and visualizes the output.

To run the pipeline:

1.  **Ensure your virtual environment is activated.**
2.  **Navigate to the `scripts` directory (optional, but paths in the script are relative to its location):**
    ```bash
    cd scripts
    ```
3.  **Execute the pipeline script:**
    ```bash
    python pipeline.py
    ```
    If you are in the root directory of the project, you can run it as:
    ```bash
    python scripts/pipeline.py
    ```

A PyVista window should open to display the colorized point cloud. Close the PyVista window to allow the script to terminate.

---

Fighting climate change, one point cloud at a time! ✨
