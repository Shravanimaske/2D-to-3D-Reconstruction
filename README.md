# 2D-to-3D-Reconstruction

An AI-powered 3D reconstruction pipeline that transforms 2D RGB images and depth information into textured 3D models.

The project aims to reconstruct indoor environments and generate 3D assets in GLB format using computer vision, depth estimation, camera pose optimization, and 3D geometry processing.

## 🚀 Project Overview

Reconstructing a 3D environment from 2D images is an important challenge in computer vision.

This project combines image processing, depth estimation, camera information, and 3D reconstruction techniques to generate realistic 3D representations of indoor scenes.

**Main objectives:**
- Convert 2D images into 3D scene representations.
- Estimate depth information from RGB images.
- Process camera poses and scene geometry.
- Generate 3D point clouds and meshes.
- Apply textures to reconstructed 3D surfaces.
- Export the final reconstruction as a `.glb` model.

## ✨ Key Features

- RGB image processing
- Depth estimation
- Camera pose processing and optimization
- 3D point cloud generation
- Geometric reconstruction
- Mesh generation and refinement
- Texture baking
- GLB model export
- Modular reconstruction pipeline

## 🏗️ Reconstruction Pipeline

The project follows these major stages:

1. **Data Loading:** Load RGB images, depth maps, and camera information.
2. **Depth Estimation:** Estimate or process depth information for scene reconstruction.
3. **Camera Pose Processing:** Estimate and refine camera positions and orientations.
4. **Point Cloud Generation:** Convert depth information into 3D point clouds.
5. **3D Geometry Reconstruction:** Combine 3D information to generate scene geometry.
6. **Mesh Processing:** Refine and clean the generated 3D mesh.
7. **Texture Mapping:** Apply image textures to the reconstructed geometry.
8. **3D Model Export:** Export the final textured model in GLB format.

## 🛠️ Technology Stack

- **Programming Language:** Python
- **Computer Vision:** OpenCV
- **Deep Learning:** PyTorch
- **3D Processing:** Open3D
- **Camera Reconstruction:** COLMAP
- **Image Processing:** NumPy
- **3D Output:** GLB

*Note: The technologies listed should be adjusted to match the packages and modules actually used in the project.*

## 📁 Project Structure

```text
2D-to-3D-Reconstruction/
│
├── src/
│   ├── modules/
│   └── ...
│
├── tests/
│
├── pipeline.py
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Shravanimaske/2D-to-3D-Reconstruction.git
```

### 2. Navigate to the project

```bash
cd 2D-to-3D-Reconstruction
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Running the Project

Run the main reconstruction pipeline:

```bash
python pipeline.py
```

The pipeline processes the configured input data and generates reconstructed 3D output.

## 📦 Output

The intended output is a textured 3D model in GLB format.

GLB files can be viewed using:
- Blender
- Windows 3D Viewer (where supported)
- Three.js-based 3D viewers
- Online GLB viewers

## 🧪 Testing

Run the test suite:

```bash
python -m pytest tests -v
```

## 🎯 Project Goals

- Improve 3D reconstruction accuracy.
- Generate realistic indoor scene geometry.
- Minimize manual parameter adjustments.
- Improve camera alignment and depth consistency.
- Produce textured and usable 3D models.

## 🔮 Future Improvements

- Improved depth estimation
- Better camera pose optimization
- Advanced mesh refinement
- Improved texture quality
- Reconstruction quality evaluation
- Support for additional indoor environments

## 👩‍💻 Author

**Shravani Maske**

GitHub: [@Shravanimaske](https://github.com/Shravanimaske)

---

⭐ If you find this project interesting, consider giving the repository a star!
