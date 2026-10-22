# Unit 3: Camera Calibration
> **Comprehensive Study Guide & Question Bank for MTE**  
> *Subject Code:* BTCS301T | *Matches Syllabus in:* [computer_vision.txt](file:///home/akapandey/Learning/computer-vision/computer_vision.txt)

---

## Table of Contents
1. [Overview of Camera Calibration](#1-overview-of-camera-calibration)
2. [Hierarchy of Camera Models](#2-hierarchy-of-camera-models)
   - *Full Perspective vs. Weak Perspective vs. Orthographic vs. Affine*
3. [Intrinsic & Extrinsic Parameters (The Projection Matrix)](#3-intrinsic--extrinsic-parameters)
   - *Intrinsic Matrix $K$ (Focal Length, Principal Point, Skew)*
   - *Extrinsic Parameters ($R$ and $t$)*
   - *The Unified $3 \times 4$ Camera Projection Equation*
4. [Radial & Tangential Lens Distortion](#4-radial--tangential-lens-distortion)
   - *Barrel Distortion vs. Pincushion Distortion*
   - *Mathematical Distortion Correction Model*
5. [Calibration Methods](#5-calibration-methods)
   - *Direct Linear Transform (DLT)*
   - *Zhang's Planar Checkerboard Method*
   - *Why a Single Image is Insufficient*
6. [Decomposing Projection Matrix $P$ (RQ / QR Decomposition)](#6-decomposing-projection-matrix-p)
7. [Reprojection Error (The Calibration Quality Metric)](#7-reprojection-error)
   - *Why More Images Yield Better Optimization*
8. [Quick-Review Exam Summary & MCQ Answer Keys](#8-quick-review-exam-summary--mcq-answer-keys)

---

## 1. Overview of Camera Calibration

### Intuitive Explanation
A camera is fundamentally a mathematical mapping machine: it takes a 3D point in the real world $(X, Y, Z)$ and projects it onto a 2D sensor pixel $(u, v)$.
- But where is the camera sitting in the room? (Position and angle $\implies$ **Extrinsic Parameters**).
- What lens is attached? What is its zoom, sensor resolution, and optical center? $\implies$ **Intrinsic Parameters**).
- Does the glass lens curve straight lines like a fish-eye? $\implies$ **Lens Distortion**).

> **Camera Calibration**: The process of mathematically estimating the intrinsic internal optical parameters, extrinsic external position/orientation parameters, and lens distortion coefficients of a camera system.

---

## 2. Hierarchy of Camera Models
*(Asked in Tutorial 3 Q7, Make up Class 1 Q4, Make up Class 5 Q4, MCQ Q49, Q50, Q60)*

To mathematically project 3D points onto a 2D plane, computer vision uses several camera models depending on distance, lens type, and accuracy requirements:

```
                  =============================================
                  COMPARING CAMERA PROJECTION MODELS
                  =============================================

    Full Perspective                   Weak Perspective                 Orthographic
    ----------------                   ----------------                 ------------
         Image                             Image                           Image
         Plane                             Plane                           Plane
           |                                 |                               |
    \      |      /                  ||      |      ||               ||      |      ||
     \     |     /                   ||      |      ||               ||      |      ||
      \    |    /                    ||      |      ||               ||      |      ||
       \   |   /                     ||      |      ||               ||      |      ||
        \  |  /                      ||      |      ||               ||      |      ||
         \ | /                       ||      |      ||               ||      |      ||
          \|/ Optical Center          |      |       |                |      |       |
           O                          <-- f/Z_bar -->                 (Parallel Rays)
    (Rays Converge to O)             (Scaled Parallel)               (True Parallel)
```

| Camera Model | How it Works Mathematically | Valid Conditions / Assumptions | Real-World Scenario *(Tutorial 3 Q7)* |
| :--- | :--- | :--- | :--- |
| **Full Perspective** | Rays converge to a single optical center $O$. True non-linear projection: $u = f \frac{X}{Z}, v = f \frac{Y}{Z}$. Accounts for **depth-dependent foreshortening** *(MCQ Q60)*. | Universal standard model for physical cameras with finite focal length. | **Handheld smartphone camera close to a subject**, autonomous vehicle dashcam. |
| **Weak Perspective (Scaled Orthographic)** | Projects 3D points orthographically onto a mean depth plane $\bar{Z}$, then applies uniform perspective scaling $m = \frac{f}{\bar{Z}}$. | Valid when **depth variation across object is tiny relative to viewing distance**: $\mathbf{\Delta Z \ll \bar{Z}}$ *(MCQ Q50)*, and field of view is small. | **Overhead warehouse camera mounted 6m above shelves** inspecting small boxes; telephoto portraits. |
| **Orthographic (Parallel)** | Rays are completely parallel and perpendicular to image sensor plane ($f \to \infty, Z \to \infty$). Drops $Z$ coordinate: $u = X, v = Y$. **Discards depth effects completely** *(MCQ Q49)*. | Distance to object is practically infinite; no magnification changes. | **Satellite earth observation imaging**, architectural drawings, telecentric machine-vision inspection. |
| **Affine Camera** | Generalizes weak perspective and orthographic models into a linear $2 \times 4$ projection matrix. Preserves parallelism and affine ratios. | Linear approximation across moderate viewing angles. | Fast feature tracking in video sequences where perspective non-linearities can be ignored. |

---

## 3. Intrinsic & Extrinsic Parameters

### The Coordinate Chain
To map a point from the physical world into an image pixel, we traverse 3 coordinate frames:
$$\text{World Coordinates } (X_w, Y_w, Z_w) \xrightarrow[\text{Extrinsics}]{\mathbf{[R \mid t]}} \text{Camera Frame } (X_c, Y_c, Z_c) \xrightarrow[\text{Intrinsics}]{\mathbf{K}} \text{Pixel Coordinates } (u, v)$$

---

### A. Extrinsic Parameters ($[R \mid t]$)
*(Asked in Tutorial 3 Q3, Make up Class 5 Q2, MCQ Q42)*

* **Definition**: Parameters that define the 3D position and orientation of the camera coordinate system relative to a fixed world coordinate system.
* **Composed of**:
  1. **Rotation Matrix ($R$)** ($3 \times 3$): 
     - 3 degrees of freedom (Euler angles: yaw, pitch, roll).
     - Orthonormal matrix: $R^T R = I$ and $\det(R) = +1$.
     - Rotates world axes to align with camera optical axes.
  2. **Translation Vector ($t$)** ($3 \times 1$):
     - 3 degrees of freedom ($t_x, t_y, t_z$).
     - Represents the position of the world origin relative to the camera center: $t = -R C$ (where $C$ is camera center in world coordinates).
* **Physical Example *(Tutorial 3 Q3)*:** For a camera mounted on a robot arm, $R$ defines the gripper orientation (pointing down, tilted, rotated), and $t$ specifies the 3D displacement of the camera from the robot's base origin.

$$\begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix} = R \begin{bmatrix} X_w \\ Y_w \\ Z_w \end{bmatrix} + t = \begin{bmatrix} R & t \end{bmatrix} \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}$$

---

### B. Intrinsic Parameters (Matrix $K$)
*(Asked in Tutorial 3 Q2, Make up Class 5 Q1, MCQ Q41, Q43, Q54, Q55)*

* **Definition**: Internal geometric and optical characteristics of the camera that map metric 3D camera coordinates $(X_c, Y_c, Z_c)$ to dimensionless 2D pixel coordinates $(u, v)$.
* **Matrix Structure ($3 \times 3$ Upper Triangular Matrix $K$)**:
  $$\mathbf{K = \begin{bmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}}$$

#### Physical Meaning of Each Intrinsic Parameter:
1. **Focal Length in Pixels ($f_x, f_y$)** *(MCQ Q43)*:
   - Physical focal length $f$ (in mm) scaled by pixel density on the sensor:
     $$f_x = f \cdot m_x = \frac{f}{p_x}, \quad f_y = f \cdot m_y = \frac{f}{p_y}$$
   - $m_x, m_y$ are number of pixels per millimeter; $p_x, p_y$ are physical pixel dimensions in mm.
   - If pixels are square, $f_x = f_y$.
2. **Principal Point Coordinates $(c_x, c_y)$** *(MCQ Q54)*:
   - The pixel location where the **optical axis perpendicularly intersects the image sensor plane**.
   - Ideally located at the geometric center of the sensor (e.g., $(960, 540)$ on a $1920 \times 1080$ sensor), but shifts due to lens mounting misalignments.
3. **Skew Coefficient ($s$)** *(MCQ Q55)*:
   - Accounts for any **shear or non-perpendicularity** between the sensor pixel grid axes (x-axis and y-axis).
   - In modern CMOS/CCD digital cameras, fabrication is nearly perfect, so $\mathbf{s \approx 0}$.

---

### C. The Full Camera Projection Matrix ($P$)
*(Asked in Tutorial 3 Q1, Make up Class 5 Q5, MCQ Q51)*

Combining intrinsic and extrinsic parameters yields the complete $3 \times 4$ projection matrix $P$:
$$\mathbf{P = K \begin{bmatrix} R & t \end{bmatrix}}$$

$$\begin{bmatrix} u \cdot w \\ v \cdot w \\ w \end{bmatrix} = \begin{bmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} r_{11} & r_{12} & r_{13} & t_x \\ r_{21} & r_{22} & r_{23} & t_y \\ r_{31} & r_{32} & r_{33} & t_z \end{bmatrix} \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}$$

* **Matrix Dimensions:** $3 \times 4$ *(MCQ Q51)*.
* **Degrees of Freedom (DOF):** 
  - 5 Intrinsic ($f_x, f_y, c_x, c_y, s$)
  - 6 Extrinsic (3 Rotation angles + 3 Translation coordinates)
  - **Total = 11 Degrees of Freedom** (up to an arbitrary global scale factor).
* To convert from homogeneous image coordinates to actual pixel values:
  $$\mathbf{u = \frac{u \cdot w}{w}, \quad v = \frac{v \cdot w}{w}}$$

---

## 4. Radial & Tangential Lens Distortion

### Intuitive Explanation
In theoretical pinhole geometry, straight real-world lines project to straight image lines. But physical glass lenses are curved and introduce optical distortions:
- Rays traveling near the outer edges of a curved lens bend differently than rays passing through the center.
- As a result, straight lines in the world appear **curved** in the photograph *(MCQ Q44)*!

```
      Pincushion Distortion             Undistorted Grid              Barrel Distortion
        (Positive k₁ > 0)                 (Ideal Lens)                (Negative k₁ < 0)
        +---------------+               +---------------+             +---------------+
        | \           / |               | | | | | | | | |             |  /---------\  |
        |  |         |  |               | | | | | | | | |             | /           \ |
        |  |         |  |               | | | | | | | | |             ||             ||
        |  |         |  |               | | | | | | | | |             ||             ||
        | /           \ |               | | | | | | | | |             | \           / |
        +---------------+               +---------------+             +---------------+
       (Bows Inward to Center)            (Straight Lines)           (Bows Outward from Center)
          [Telephoto Lenses]                                            [Wide-Angle Lenses]
```

---

### Comparison of Distortion Types
*(Asked in Tutorial 3 Q4, Make up Class 5 Q3, MCQ Q45, Q46, Q56)*

| Distortion Type | Geometric Effect | Distortion Coefficient | Common Lens Types | Physical Cause |
| :--- | :--- | :---: | :--- | :--- |
| **Barrel Distortion** | Lines bow **outward away from the center** *(MCQ Q45)*; magnification decreases with radial distance. | $k_1 < 0$ | **Wide-angle lenses**, GoPros, fisheye lenses, drone cameras. | Field of view is larger than sensor size; edges compress inward. |
| **Pincushion Distortion** | Lines bow **inward toward the center** *(MCQ Q46)*; magnification increases with radial distance. | $k_1 > 0$ | **Telephoto lenses**, high-zoom cameras. | Peripheral rays are magnified more than central rays. |
| **Tangential Distortion** | Non-radial asymmetry (elliptical warping). | $p_1, p_2$ | Inexpensive or poorly assembled webcams/phone cameras. | **Lens and sensor plane are not perfectly parallel** (decentering error) *(MCQ Q56)*. |

---

### Mathematical Distortion Model
Let $(x, y)$ be normalized undistorted coordinates, and $r^2 = x^2 + y^2$ be radial distance from the principal point:

1. **Radial Distortion Correction Formula**:
   $$\mathbf{x_{distorted} = x \left( 1 + k_1 r^2 + k_2 r^4 + k_3 r^6 \right)}$$
   $$\mathbf{y_{distorted} = y \left( 1 + k_1 r^2 + k_2 r^4 + k_3 r^6 \right)}$$
2. **Tangential Distortion Correction Formula**:
   $$x_{tangential} = 2 p_1 x y + p_2 (r^2 + 2x^2)$$
   $$y_{tangential} = p_1 (r^2 + 2y^2) + 2 p_2 x y$$

---

## 5. Calibration Methods

### A. Direct Linear Transform (DLT)
*(Asked in MCQ Q58)*

* **Concept**: Uses a known 3D calibration target (e.g., two perpendicular planes with known 3D grid points $X_w$).
* **Mathematical Setup**:
  For each 3D point $X_i = [X_i, Y_i, Z_i, 1]^T$ and its observed 2D pixel $x_i = [u_i, v_i, 1]^T$, the cross product is zero: $x_i \times (P X_i) = 0$.
  This expands into 2 linearly independent equations per point:
  $$\begin{bmatrix} X_i^T & \mathbf{0}^T & -u_i X_i^T \\ \mathbf{0}^T & X_i^T & -v_i X_i^T \end{bmatrix} \begin{bmatrix} p_1 \\ p_2 \\ p_3 \end{bmatrix} = \mathbf{0}$$
* **Solving**:
  - $P$ has 11 unknown parameters (12 entries up to global scale).
  - Each point provides 2 equations $\implies$ **We need at least 6 non-coplanar 3D-to-2D point correspondences** ($6 \times 2 = 12$ equations).
  - Solved using **Singular Value Decomposition (SVD)**: $P$ is the eigenvector corresponding to the smallest eigenvalue of $A^T A$.

---

### B. Zhang's Planar Calibration Method (Checkerboard Calibration)
*(Asked in Tutorial 3 Q5, Assignment 3, MCQ Q47, Q52)*

The industry-standard camera calibration technique (used in OpenCV, MATLAB, ROS).

```
         Checkerboard at Orientation 1       Checkerboard at Orientation 2
                   +---+---+                           +---+---+
                   | X |   |                           |   | X |
                   +---+---+                           +---+---+
                     \                                     /
                      \                                   /
                       \                                 /
                        \                               /
                         ▼                             ▼
                              [ CAMERA UNDER TEST ]
```

#### Step-by-Step Procedure:
1. **Prepare Planar Target**: Print a high-contrast checkerboard pattern of known square size (e.g., $30 \times 30$ mm).
2. **Acquire Multi-View Photos**: Capture 10 to 20 images of the board held at varying angles, tilts, and distances across the camera's field of view.
3. **Detect Feature Corners**: Use corner detection (Harris / Shi-Tomasi) to find internal checkerboard saddle points with sub-pixel precision.
4. **Compute Homography per Image**:
   - Because the board is planar, we set $Z_w = 0$.
   - The projection simplifies to a $3 \times 3$ **Planar Homography** $H = \begin{bmatrix} h_1 & h_2 & h_3 \end{bmatrix} = \lambda K \begin{bmatrix} r_1 & r_2 & t \end{bmatrix}$.
   - We solve $H$ for each photo from the known grid points.
5. **Solve for Intrinsic Matrix $K$**:
   - Using the orthonormality of rotation vectors ($r_1^T r_2 = 0$ and $\|r_1\| = \|r_2\|$), each image homography provides **2 linear constraints** on the absolute conic matrix $B = K^{-T} K^{-1}$.
   - Since $B$ has 5 unknown parameters, at least **3 distinct views** are required to solve for $K$ analytically.
6. **Extract Extrinsics & Refine via Non-Linear Optimization**:
   - Calculate $R$ and $t$ for each image.
   - Run **Levenberg-Marquardt optimization** to jointly refine $K, R, t, k_1, k_2$ by minimizing total reprojection error.

---

### ❓ Why Can't a Single Image Alone Calibrate a Camera?
*(Direct Question from Tutorial 3 Q10c)*
1. **Mathematical Under-determination**: A single planar view yields only one homography $H$, giving **2 independent constraints**. But the camera intrinsic matrix $K$ has **5 unknown parameters** ($f_x, f_y, c_x, c_y, s$)! You cannot solve for 5 unknowns using only 2 equations.
2. **Projective Ambiguity**: From a single 2D image, there is an infinite family of 3D camera orientations and focal lengths that produce the exact same 2D projection (focal length and object distance scale proportionally). Multiple views from different angles break this ambiguity!

---

## 6. Decomposing Projection Matrix $P$ (RQ / QR Decomposition)
*(Asked in Tutorial 3 Q6, MCQ Q53)*

Given an estimated $3 \times 4$ projection matrix $P$:
$$P = \begin{bmatrix} M & | & p_4 \end{bmatrix}$$
where $M$ is the left $3 \times 3$ submatrix:
$$M = K R$$

### The Logic Behind It
- $K$ is an **upper-triangular matrix** (due to camera pinhole geometry).
- $R$ is an **orthogonal matrix** ($R R^T = I$, pure rotation).
- Any invertible square matrix can be uniquely factored into the product of an Upper-triangular matrix and an Orthogonal matrix. This is called **RQ Decomposition** *(MCQ Q53)*!

### The Extraction Formulas
1. Perform RQ decomposition on $M$:
   $$M = K \cdot R$$
   - Gives intrinsic matrix $K$ directly (scaled so $K_{33} = 1$).
   - Gives orthogonal rotation matrix $R$ directly.
2. Recover Translation vector $t$:
   $$p_4 = K \cdot t \implies \mathbf{t = K^{-1} p_4}$$
3. Recover 3D Camera Center in World Coordinates ($C$):
   $$P \begin{bmatrix} C \\ 1 \end{bmatrix} = \mathbf{0} \implies M C + p_4 = 0 \implies \mathbf{C = -M^{-1} p_4 = -R^T t}$$

---

## 7. Reprojection Error

### Definition
*(Asked in Tutorial 3 Q8, Assignment 3 Q5, MCQ Q48)*

> **Reprojection Error**: The geometric distance (measured in pixels) between an actual observed 2D corner point $x_i$ detected in the image and the estimated 2D point $\hat{x}_i = P X_i$ computed by projecting the 3D target point through the estimated camera parameters.

### Formula to Remember
$$\mathbf{\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N \| x_i - \hat{x}_i \|^2} = \sqrt{\frac{1}{N} \sum_{i=1}^N \left( (u_i - \hat{u}_i)^2 + (v_i - \hat{v}_i)^2 \right)}}$$

```
                Detected Corner x_i (Observed)
                      *
                     /
                    /  <--- Euclidean Reprojection Error (Distance in pixels)
                   /
                  * 
                Projected Model Point x̂_i = P * X_i
```

* **Interpretation**:
  - $\text{RMSE} < 0.5$ pixels: **Excellent calibration**.
  - $\text{RMSE} > 1.5$ pixels: **Poor calibration** (indicates blurry photos, bad corner detection, or wrong distortion modeling).

---

### Why More Images Improve Calibration Accuracy
*(Direct Question from Tutorial 3 Q9, MCQ Q57)*

1. **Over-determined Least-Squares Optimization**:
   Camera calibration is solved using non-linear least squares (Levenberg-Marquardt). Using 15 images instead of 5 provides $3 \times$ more independent geometric constraints, preventing the optimizer from falling into local minima or overfitting to corner noise.
2. **Sensor-Wide Lens Distortion Coverage**:
   Lens distortion (especially barrel distortion) is weakest at the image center and strongest at the extreme sensor corners. Capturing 15 diverse images ensures checkerboard corners occupy all 4 periphery quadrants of the sensor, allowing distortion coefficients ($k_1, k_2$) to be fitted accurately.
3. **Cancellation of Random Noise**:
   Corner detection algorithms have sub-pixel jitter due to lighting and sensor noise. Fitting parameters over hundreds of corners across 15 views averages out zero-mean Gaussian jitter!

---

## 8. Quick-Review Exam Summary & MCQ Answer Keys

### Mnemonic / Memory Cheat Sheet for MTE
- **Projection Matrix $P$**: Dimensions are $\mathbf{3 \times 4}$, Degrees of freedom = **11**.
- **Intrinsic Matrix $K$**: Upper triangular ($f_x, f_y, c_x, c_y, s$) $\to$ internal optics $\to$ recovered via **RQ decomposition**.
- **Extrinsic Parameters**: $[R \mid t]$ $\to$ 3D rotation and translation $\to$ camera pose in world.
- **Radial Distortion**:
  - Straight lines become **curved**.
  - **Barrel** ($k_1 < 0$): Bows **outward** (Wide-angle/GoPro).
  - **Pincushion** ($k_1 > 0$): Bows **inward** (Telephoto/Zoom).
  - **Tangential**: Lens not parallel to sensor (decentering).
- **Weak Perspective Condition**: Depth variation is tiny compared to distance: $\mathbf{\Delta Z \ll \bar{Z}}$.
- **Orthographic**: Parallel rays $\to$ **discards depth effects**.
- **Calibration Target**: **Checkerboard pattern** (Zhang's method) $\to$ requires at least **3 views** to mathematically constrain $K$.
- **Quality Metric**: **Reprojection error** (Euclidean pixel distance between detected and projected points).

---

### Official MCQ Solutions (Assignment 9 / Unit 3 Questions 41–60)
| Q# | Question Summary | Correct Option | Key Rationale |
| :---: | :--- | :---: | :--- |
| **Q41** | Intrinsic camera parameters typically include: | **B. Focal length, principal point, and skew** | The 5 upper-triangular elements of $K$. |
| **Q42** | Extrinsic camera parameters describe: | **B. Camera's rotation and translation relative to world** | Coordinate transform from world to camera frame. |
| **Q43** | In calibration, focal length is often expressed in: | **B. Pixels** | Scaled by pixel sensor density ($f_x = f \cdot m_x$). |
| **Q44** | Radial lens distortion causes straight real-world lines to: | **B. Curved** | Non-linear bending of peripheral optical rays. |
| **Q45** | "Barrel" distortion is characterised by lines that: | **B. Bow outward away from the centre** | Wide-angle lens compression at image boundaries. |
| **Q46** | "Pincushion" distortion is characterised by lines that: | **A. Bow inward toward the centre** | Telephoto magnification expansion at edges. |
| **Q47** | A commonly used calibration target for parameters: | **B. Checkerboard pattern** | High-contrast planar saddle corners. |
| **Q48** | Reprojection error measures difference between: | **B. Observed image points & reprojected estimated positions**| Gold standard calibration quality residual. |
| **Q49** | Orthographic projection model essentially discards: | **B. Perspective/depth effects** | Rays are parallel; magnification is constant. |
| **Q50** | Weak-perspective model is a reasonable approximation when: | **B. Depth variation is small relative to viewing distance** | Object thickness $\Delta Z \ll \bar{Z}$. |
| **Q51** | General camera projection matrix $P$ has dimensions: | **C. 3×4** | Maps homogeneous 4-vector 3D point to 3-vector 2D pixel. |
| **Q52** | Zhang's calibration method requires: | **B. Multiple images of planar pattern at different orientations**| Homography decomposition from $\ge 3$ views. |
| **Q53** | Intrinsic & extrinsic parameters recovered from $P$ using: | **A. RQ (or QR) decomposition** | Decomposes $M = K R$ into triangular $\times$ orthogonal. |
| **Q54** | The principal point of a camera refers to: | **B. Intersection of optical axis with image plane** | Optical center pixel on the sensor. |
| **Q55** | Skew parameter in intrinsic matrix accounts for: | **B. Non-perpendicularity between pixel axes** | Shear angle between sensor x and y pixel grid. |
| **Q56** | Tangential distortion arises due to: | **B. Lens and sensor plane not being parallel (decentering)**| Physical misalignment of optical elements. |
| **Q57** | Using more calibration images generally: | **B. Reduces reprojection error & improves accuracy** | Overdetermined non-linear optimization cancels noise. |
| **Q58** | Direct Linear Transform (DLT) is used to: | **A. Estimate projection matrix from 3D–2D correspondences**| Solves $A p = 0$ via SVD using $\ge 6$ points. |
| **Q59** | Camera intrinsic matrix $K$ primarily encodes: | **B. Camera-internal parameters like focal length & principal point**| Hardware optical properties. |
| **Q60** | Compared to orthographic, full perspective additionally accounts for: | **B. Depth-dependent foreshortening** | Farther objects project smaller ($1/Z$ scaling). |
