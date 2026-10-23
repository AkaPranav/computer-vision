# Unit 2: Feature Detection & Segmentation
> **Comprehensive Study Guide & Question Bank for MTE**  
> *Subject Code:* BTCS301T | *Matches Syllabus in:* [computer_vision.txt](file:///home/akapandey/Learning/computer-vision/computer_vision.txt)

---

## Table of Contents
1. [Overview of Features in Computer Vision](#1-overview-of-features-in-computer-vision)
2. [Edge Detection & The Canny Algorithm](#2-edge-detection--the-canny-algorithm)
   - *Image Gradients (Sobel/Prewitt)*
   - *Canny 4-Step Algorithm (Deep Dive)*
3. [Corner Detection & The Harris Detector](#3-corner-detection--the-harris-detector)
   - *Why Corners > Edges (Aperture Problem)*
   - *Structure Tensor & Harris Response Matrix Math*
4. [Line & Curve Detection (Hough Transform)](#4-line--curve-detection-hough-transform)
5. [Active Contours (Snakes)](#5-active-contours-snakes)
6. [Feature Descriptors (SIFT, HOG, Shape Context)](#6-feature-descriptors)
   - *SIFT: 4 Stages to Scale & Rotation Invariance*
   - *HOG: Gradients, Cells & Pedestrian Detection*
   - *Shape Context: Log-Polar Geometry*
7. [Morphological Operations (Erosion, Dilation, Opening, Closing)](#7-morphological-operations)
   - *Worked Example with Structuring Element*
8. [Image Segmentation Algorithms](#8-image-segmentation-algorithms)
   - *Split-and-Merge (Quadtree)*
   - *Watershed Algorithm & Marker-Based Solution*
   - *Mean-Shift Mode Seeking*
   - *Graph-Based Segmentation & Normalized Cut (N-Cut)*
9. [Quick-Review Exam Summary & MCQ Answer Keys](#9-quick-review-exam-summary--mcq-answer-keys)

---

## 1. Overview of Features in Computer Vision

### Intuitive Explanation
If you want to stitch two photos together to make a panorama, how does the computer know where they overlap?
- Looking at every single pixel is slow and easily corrupted by lighting changes.
- Instead, the computer searches for **distinctive landmarks** called **features** (edges, corners, distinctive patterns) that are easy to locate and match across photos.

```text
+-------------------------------------------------------------+
| Region Type | Can We Localize Position? | Example In Photo  |
+-------------------------------------------------------------+
| Flat        | No (Ambiguous in all directions)| Clear blue sky   |
| Edge        | Ambiguous along the edge line   | Horizon / Wall edge|
| Corner      | YES (Constrained in both x & y) | Table corner, window |
+-------------------------------------------------------------+
```

---

## 2. Edge Detection & The Canny Algorithm

### What is an Edge?
An edge is a significant local change in image intensity, typically occurring at boundaries between different objects or surface markings.

### Mathematical Formulation: The Gradient Vector
The intensity gradient of an image I(x, y) is a 2D vector pointing in the direction of the steepest intensity increase:

```text
∇I = [ Ix ] = [ ∂I/∂x ]
     [ Iy ]   [ ∂I/∂y ]
```

1. **Gradient Magnitude** (Strength of the edge):
   ```text
   |∇I| = sqrt( Ix² + Iy² ) ≈ |Ix| + |Iy|
   ```
2. **Gradient Direction** (Perpendicular to edge direction):
   ```text
   θ = arctan( Iy / Ix )
   ```

* **Sobel Kernels** (used to estimate derivatives Ix, Iy):
  ```text
  Sx (Horizontal Gradient):       Sy (Vertical Gradient):
  [ -1   0  +1 ]                  [ +1  +2  +1 ]
  [ -2   0  +2 ]                  [  0   0   0 ]
  [ -1   0  +1 ]                  [ -1  -2  -1 ]
  ```

---

### 🌟 The Canny Edge Detector (The Gold Standard)
*(Asked in Tutorial 2 Q1, Make up Class 4 Q1, MCQ Q21, Q22)*

John F. Canny formulated edge detection as an optimal trade-off between three criteria:
1. **Low error rate**: Catch all true edges and produce no spurious responses.
2. **Good localization**: Distance between detected edge pixel and true edge is minimized.
3. **Single response**: No multiple responses to a single edge (thin 1-pixel edges).

#### The 4 Main Steps (Must Memorize in Order):

```text
Input Image 
    │
    ▼
[Step 1: Gaussian Smoothing]               --> Suppresses noise
    │
    ▼
[Step 2: Gradient Computation (Sobel)]     --> Finds magnitude & angle θ
    │
    ▼
[Step 3: Non-Maximum Suppression (NMS)]    --> Thins edges to 1-pixel width
    │
    ▼
[Step 4: Hysteresis Thresholding]          --> Resolves broken edges with two thresholds
    │
    ▼
Clean, Thin Binary Edge Map
```

1. **Step 1: Gaussian Filtering**
   - Convolves the raw image with a Gaussian kernel to remove high-frequency noise that could otherwise trigger false edge detections.
2. **Step 2: Finding Intensity Gradients**
   - Applies Sobel masks to find horizontal (Ix) and vertical (Iy) derivatives.
   - Computes gradient magnitude M(x, y) and direction θ(x, y).
   - Quantizes θ into 4 canonical directions: 0° (horizontal), 45° (diagonal), 90° (vertical), 135° (anti-diagonal).
3. **Step 3: Non-Maximum Suppression (NMS)** *(MCQ Q22)*
   - Raw gradient magnitudes produce thick, blurred ridges along edges.
   - **The Logic**: At every pixel, compare its magnitude with its two immediate neighbors along the gradient direction θ. If the pixel is **not strictly larger** than both neighbors, set it to 0.
   - **Result**: Thins the thick edge response down to a crisp, **single-pixel width**.
4. **Step 4: Double Thresholding & Hysteresis Tracking**
   - Uses two thresholds: a high threshold (T_high) and a low threshold (T_low, typically T_low ≈ 0.4 to 0.5 × T_high).
   - **Strong Edges** (M > T_high): Kept unconditionally.
   - **Suppressed Non-Edges** (M < T_low): Immediately rejected as noise.
   - **Weak Edges** (T_low <= M <= T_high): **Kept if and only if** they are connected via an 8-connected chain to a **strong edge**. This prevents broken dashed lines along true object boundaries!

---

## 3. Corner Detection & The Harris Detector

### Why Corners are More Informative than Edges
*(Asked in Tutorial 2 Q2, MCQ Q23-Q24)*

#### The Aperture Problem
Look at an image through a tiny circular window (aperture):
- **Flat region**: Moving the window in any direction yields zero change in intensity.
- **Straight edge**: Moving the window along the direction of the edge yields **no change in intensity**! The motion along the edge line is fundamentally ambiguous.
- **Corner**: Moving the window in **any direction** (x or y) causes an immediate, noticeable change in pixel intensities. Hence, corners can be localized in 2D uniquely!

```text
      Flat Region                  Edge Region                 Corner Region
    +--------------+            +--------------+            +--------------+
    |              |            |     |        |            |      |       |
    |      O       |            |   O |        |            |   ---+       |
    |              |            |     |        |            |      |   O   |
    +--------------+            +--------------+            +--------------+
   Shift in any dir:           Shift along edge:           Shift in any dir:
      NO CHANGE                   NO CHANGE                LARGE INTENSITY CHANGE!
```

---

### The Harris Corner Detector Formulation

#### 1. Logic Behind the Math
We measure the change in intensity E(u, v) when shifting an image window W by an offset (u, v):
```text
E(u, v) = sum_{window} w(x, y) * [ I(x+u, y+v) - I(x, y) ]²
```
Using first-order Taylor series approximation: `I(x+u, y+v) ≈ I(x, y) + u·Ix + v·Iy`, this simplifies to a quadratic matrix form:
```text
E(u, v) ≈ [ u  v ] · M · [ u ]
                         [ v ]
```

#### 2. The Structure Tensor (Second Moment Matrix M)
```text
M = sum_{window} w(x, y) * [ Ix²      Ix·Iy ]
                           [ Ix·Iy    Iy²   ]
```
The eigenvalues of M, denoted **λ₁** and **λ₂**, describe the shape of the local intensity surface:

```text
        Eigenvalue Analysis of Matrix M
        =================================
        λ₂ ^
           |       Edge
           |   (λ₂ large, λ₁ small)      CORNER (Both λ₁ & λ₂ LARGE)
           |
           |       Flat
           |   (Both λ₁, λ₂ small)       Edge (λ₁ large, λ₂ small)
           +--------------------------------------------------------> λ₁
```

| Region Type | Eigenvalue Behavior | Physical Meaning |
| :--- | :--- | :--- |
| **Flat Region** | λ₁ ≈ 0, λ₂ ≈ 0 | No gradient in any direction |
| **Edge Region** | One λ large, one λ ≈ 0 | Gradient in one direction, flat in perpendicular |
| **Corner Region** | **Both λ₁ and λ₂ are large** *(MCQ Q24)* | Significant gradient in multiple directions |

#### 3. Harris Corner Response Function (R)
Computing explicit eigenvalues for every single pixel is slow. Harris and Stephens proposed an algebraic shortcut using determinant and trace:

> **Formula to Remember:**
> ```text
> R = det(M) - k · (trace(M))²
> R = (λ₁ · λ₂) - k · (λ₁ + λ₂)²
> ```
> * **det(M)** = λ₁ · λ₂ = (M11 · M22 - M12²)
> * **trace(M)** = λ₁ + λ₂ = (M11 + M22)
> * **k** is an empirical sensitivity constant (typically k = 0.04 to 0.06).

* **Classification based on R:**
  - **R > 0** (large positive): **Corner**
  - **R < 0** (large negative): **Edge**
  - **|R| ≈ 0** (close to zero): **Flat region**

---

## 4. Line & Curve Detection (Hough Transform)

### Intuitive Explanation
If you have scattered edge points from a Canny detector, how do you know which ones lie on a straight line?
- Testing every pair of points is slow (O(N²)) and vulnerable to missing pixels.
- The **Hough Transform** uses a "voting scheme". Every edge point casts votes for all possible lines that could pass through it. The line with the most votes wins!

### Mathematical Formulation
In slope-intercept form `y = m·x + c`, vertical lines have infinite slope (m = ∞), causing crashes. 
The Hough transform uses the **Polar (Normal) Equation of a line**:

> **Formula to Remember:**
> ```text
> ρ = x · cos(θ) + y · sin(θ)
> ```
> * **ρ (rho)**: Perpendicular distance from origin to the line.
> * **θ (theta)**: Angle of the normal vector relative to the x-axis (-90° to +90° or 0° to 180°).

```text
  Image Space (x, y)                        Hough Parameter Space (ρ, θ)
  -------------------                       ----------------------------
     y ^                                       ρ ^
       |     * (Point 1)                         |        ~~~ (Sinusoid 1)
       |       * (Point 2)                       |       /   \
       |         * (Point 3)                     |------X-----X----- (Intersection = Peak Line)
       +-------------------> x                   +-------------------------> θ
  (Collinear points)                        (Curves intersect at single point!)
```

* **Procedure**:
  1. Discretize parameter space into an accumulator 2D array `A[ρ, θ]`, initialized to zero.
  2. For every detected edge point (x_i, y_i):
     - For θ varying across allowed angles, calculate `ρ = x_i · cos(θ) + y_i · sin(θ)`.
     - Increment `A[ρ, θ] = A[ρ, θ] + 1`.
  3. Local peaks in the accumulator array correspond to dominant lines in the image!

---

## 5. Active Contours (Snakes)
*(Asked in Tutorial 2 Q10a, MCQ Q40)*

### Intuitive Explanation
Imagine stretching an elastic rubber band around an object. The rubber band wants to shrink to minimize its tension (internal force), but features like dark-to-light object boundaries act like sticky barriers pushing back against it (external force). When the forces balance, the band hugs the object's contour perfectly!

### Mathematical Formulation
A snake is a parametric curve `v(s) = [x(s), y(s)]` for parameter `s` in `[0, 1]`. It moves to **minimize total contour energy**:

```text
E_total = Integral [ E_internal(v(s)) + E_external(v(s)) ] ds
```

1. **Internal Energy (E_internal)** (Controls contour shape & smoothness):
   ```text
   E_internal = 0.5 * [ α · |v'(s)|² + β · |v''(s)|² ]
   ```
   - First derivative `|v'(s)|²`: **Elasticity / tension** (prevents stretching, pulls curve tight like rubber).
   - Second derivative `|v''(s)|²`: **Rigidity / stiffness** (prevents sharp bends, makes curve smooth).
2. **External Energy (E_external)** (Attracts snake to image boundaries):
   ```text
   E_external = -|∇I(x, y)|²
   ```
   - Negative gradient magnitude pulls the snake directly toward high-contrast object edges!

---

## 6. Feature Descriptors

### A. SIFT (Scale-Invariant Feature Transform)
*(Asked in Tutorial 2 Q4, Make up Class 3 Q2, Make up Class 4 Q2, MCQ Q25, Q26)*

* **Why SIFT is a breakthrough**: It describes visual features in a way that remains identical even if the image is **scaled**, **rotated**, or subjected to moderate **affine/illumination shifts**.
* **The 4 Sequential Stages**:
  1. **Scale-Space Extrema Detection**:
     - Convolves image with Gaussians at multiple scales σ and computes Difference of Gaussians (DoG): `D(x, y, σ) = [ G(x, y, k·σ) - G(x, y, σ) ] * I(x, y)`.
     - Each pixel is compared with its 26 space-scale neighbors (8 in current scale, 9 in scale above, 9 in scale below). Maxima/minima are candidate keypoints. ===> **Scale Invariance**.
  2. **Keypoint Localization**:
     - Fits a 3D quadratic function to locate extrema with sub-pixel precision. Discards low-contrast keypoints and unstable edge responses using the ratio of principal curvatures.
  3. **Orientation Assignment**:
     - Computes gradient magnitude and direction for pixels around the keypoint. Builds a 36-bin orientation histogram (10° per bin).
     - The peak direction is assigned as the keypoint's dominant orientation θ₀. All subsequent measurements are rotated relative to θ₀. ===> **Rotation Invariance**.
  4. **Descriptor Generation**:
     - Takes a 16 × 16 pixel neighborhood around the keypoint.
     - Divides it into a 4 × 4 grid of sub-blocks (each 4 × 4 pixels).
     - Computes an 8-bin gradient orientation histogram for each sub-block.
     - Total descriptor length = 4 × 4 × 8 = **128-dimensional vector**!
     - Normalized to unit length to provide illumination/contrast invariance.

---

### B. HOG (Histogram of Oriented Gradients)
*(Asked in Tutorial 2 Q5, Make up Class 4 Q2, MCQ Q27, Q28)*

* **Core Idea**: Local object appearance and shape can be characterized by the distribution of local intensity gradients, even without knowing exact edge positions.
* **Extraction Pipeline**:
  1. Compute 1D derivative filters: `[-1, 0, 1]` and `[-1, 0, 1]^T`.
  2. Divide image into small spatial **cells** (e.g., 8 × 8 pixels).
  3. For each cell, compile a **9-bin orientation histogram** (0° to 180° unsigned gradients), weighted by gradient magnitude.
  4. Group adjacent cells into overlapping **blocks** (e.g., 2 × 2 cells = 16 × 16 pixels).
  5. Normalize block vectors using L2-norm: `v -> v / sqrt(||v||² + ε²)` to achieve robust invariance to shadows and lighting changes.
* **Primary Application *(MCQ Q28)*:** **Pedestrian and human detection** (Dalal & Triggs, 2005).

---

### C. Shape Context Descriptors
*(Asked in Tutorial 2 Q10c, Make up Class 3 Q2)*

* **Definition**: A descriptor that captures the global geometric configuration of an object's boundary.
* **How it Works**:
  1. Sample N points along the object contour.
  2. For a specific point p_i, treat it as origin and construct a log-polar coordinate histogram (logarithmic radial distance r, uniform angle θ).
  3. Count how many of the remaining (N - 1) contour points fall into each log-polar bin.
* **Application**: Optical character recognition (OCR), silhouette recognition, trademark retrieval.

---

## 7. Morphological Operations
*(Asked in Tutorial 2 Q6, Make up Class 3 Q3, MCQ Q29-Q32)*

Morphological operations are shape-based mathematical operations performed on binary images (pixels are 0 or 1) using a small binary probe called a **Structuring Element (SE)** (e.g., a 3 × 3 cross or disk).

```text
   Structuring Element B (Cross)
             [ 0  1  0 ]
             [ 1  1  1 ]
             [ 0  1  0 ]
```

### The 4 Fundamental Operations

| Operation | Mathematical Notation | Definition / How it Works | Effect on Object | Use Case |
| :--- | :---: | :--- | :--- | :--- |
| **Erosion** | `A ⊖ B` | SE is placed at every pixel. Output is 1 **only if SE fits completely inside** foreground A. | **Shrinks object** *(MCQ Q29)*; strips outer boundary pixels | Removes thin noise whiskers, separates touching objects |
| **Dilation** | `A ⊕ B` | SE is placed at every foreground pixel. Output is 1 if SE **touches any part** of foreground A. | **Grows/expands object** *(MCQ Q30)* | Fills small holes, connects broken fragments |
| **Opening** | `A ∘ B = (A ⊖ B) ⊕ B` | **Erosion followed by Dilation** *(MCQ Q31)* | Removes small protrusions and isolated noise specks | Smoothing object contours without changing general size |
| **Closing** | `A • B = (A ⊕ B) ⊖ B` | **Dilation followed by Erosion** *(MCQ Q32)* | Fills narrow interior holes and bridges narrow gaps | Merging broken boundary segments |

---

### 🧮 Morphological Erosion Worked Example
*(Direct question from Tutorial 2 Q6)*

> **Problem:** Perform one pass of erosion (`A ⊖ B`) on the 5 × 5 binary image below using a 3 × 3 square structuring element (all 1s):
> ```text
> Binary Image A:                      Structuring Element B:
> [ 0  0  0  0  0 ]                    [ 1  1  1 ]
> [ 0  1  1  1  0 ]                    [ 1  1  1 ]
> [ 0  1  1  1  0 ]                    [ 1  1  1 ]
> [ 0  1  1  1  0 ]
> [ 0  0  0  0  0 ]
> ```

#### Working:
- Place center of B at each pixel (x, y).
- To yield 1, all 9 neighbors covered by B must be 1.
- For all border pixels of the 3 × 3 block of 1s (like (1, 1), (1, 2), etc.), at least one neighbor touches 0.
- Only for the very center pixel at coordinate (2, 2) are all 9 surrounding pixels equal to 1!

```text
Erosion Output (A ⊖ B):
[ 0  0  0  0  0 ]
[ 0  0  0  0  0 ]
[ 0  0  1  0  0 ]
[ 0  0  0  0  0 ]
[ 0  0  0  0  0 ]
```
*Notice how the 3 × 3 foreground square shrank down to a single center pixel!*

---

## 8. Image Segmentation Algorithms

### Overview
Segmentation partitions an image into disjoint, internally homogeneous regions.

---

### A. Split-and-Merge (Quadtree Segmentation)
*(Asked in Tutorial 2 Q8, Make up Class 3 Q4, Make up Class 4 Q4, MCQ Q36)*

1. **Splitting Phase**:
   - Define a homogeneity predicate P(R) (e.g., P(R) = True if standard deviation σ <= threshold).
   - Start with the entire image as one root node.
   - If P(R) = False, **split the region into 4 equal quadrants** (Quadtree structure).
   - Recursively split each child quadrant until every quadrant satisfies P(R) = True.
2. **Merging Phase**:
   - Examine adjacent quadrants R_i and R_j.
   - If they satisfy combined homogeneity P(R_i U R_j) = True, merge them into a single region.
3. **Stopping Condition**:
   - The process stops when **no further regions require splitting** (P(R_k) = True for all regions) and **no adjacent regions can be merged**.

---

### B. Watershed Segmentation
*(Asked in Tutorial 2 Q7, Make up Class 3 Q4, MCQ Q33, Q34)*

* **The Topographic Analogy**:
  - Treat a grayscale image (or its gradient magnitude) as a 3D topographic terrain:
    - High-intensity pixels / high gradients = **Mountain Ridges / Watershed Divides**.
    - Low-intensity pixels / uniform areas = **Valleys / Catchment Basins**.
  - Imagine piercing a hole at each regional local minimum and immersing the terrain in water.
  - Water floods upward from the minima. Where water from distinct basins is about to merge, we construct a dam wall.
  - When the water reaches the top, the network of dams forms the **Watershed Lines** that cleanly segment objects.

```text
       Catchment Basin 1           Watershed Dam (Edge)          Catchment Basin 2
            \     /                        ||                        \     /
             \   /                         ||                         \   /
              \_/ (Local Min 1)            ||                          \_/ (Local Min 2)
```

* **The Over-Segmentation Flaw**: Real-world images contain hundreds of microscopic texture noises, each creating its own tiny local minimum. Flooding produces a chaotic mosaic of thousands of tiny useless regions!
* **The Solution: Marker-Controlled Watershed *(MCQ Q34)*:**
  - Manually or algorithmically specify **markers**:
    1. *Internal markers*: Definite foreground objects (e.g., fruit centers).
    2. *External markers*: Definite background.
  - Water is allowed to flood ONLY from these defined marker locations, eliminating over-segmentation completely!

---

### C. Mean-Shift Segmentation
*(Asked in Tutorial 2 Q10b, MCQ Q39)*

* **Intuition**: An iterative, non-parametric clustering algorithm that seeks modes (peaks) of a probability density function.
* **Mechanism**:
  1. Map every pixel to a multi-dimensional feature space combining coordinates and colors: `(x, y, R, G, B)`.
  2. Place a spherical window (kernel) of bandwidth h around each point.
  3. Compute the center of mass (**mean**) of all data points inside the window.
  4. Shift the window center to this mean (**mean-shift vector**).
  5. Repeat until the window stops moving (converges to a local density maximum / mode).
  6. All pixels whose paths converge to the same mode are assigned to the same segment!
* **Advantage**: Requires **no prior assumption** about the number of clusters k or cluster shape.

---

### D. Graph-Based Segmentation & Normalized Cut (N-Cut)
*(Asked in Tutorial 2 Q9, Make up Class 3 Q5, MCQ Q37, Q38)*

* **Graph Representation**:
  - Image is modeled as a weighted, undirected graph G = (V, E).
  - Vertices V: Individual pixels.
  - Edges E: Connecting adjacent pixels with weight w(i, j) reflecting affinity/similarity in brightness and spatial proximity:
    ```text
    w(i, j) = exp( -||I_i - I_j||² / σ_I² ) * exp( -||x_i - x_j||² / σ_X² )
    ```

#### The Minimum Cut Problem vs. Normalized Cut
To segment the image into two groups A and B, we cut edges connecting them. The cut weight is:
```text
cut(A, B) = sum_{u in A, v in B} w(u, v)
```
* **Why Simple Min-Cut Fails**: A simple minimum cut favors separating tiny, isolated outlier pixels (since cutting 1 pixel from a large graph cuts very few edges, yielding an unnaturally small cut value!).

#### The Normalized Cut Solution (Shi & Malik)
Normalized Cut fixes this by penalizing unbalanced partitions:

> **Formula to Remember:**
> ```text
> Ncut(A, B) = [ cut(A, B) / assoc(A, V) ] + [ cut(A, B) / assoc(B, V) ]
> ```
> * **assoc(A, V)** = total connection weight from nodes in A to all nodes in the entire graph V.

* **Key Advantage *(MCQ Q38)*:** Automatically **balances segment size** while minimizing boundary similarity, avoiding isolated pixel cuts!
* Solved computationally by finding the second-smallest eigenvector of the generalized Laplacian system: `(D - W) · y = λ · D · y`.

---

## 9. Quick-Review Exam Summary & MCQ Answer Keys

### Mnemonic / Memory Cheat Sheet for MTE
- **Canny Edge Steps (GSNH)**: **G**aussian blur -> **S**obel gradient -> **N**on-maximum suppression (thinning) -> **H**ysteresis (double threshold).
- **Harris Corner**: Structure Tensor M -> Corners require **BOTH eigenvalues λ₁, λ₂ to be large** -> `R = det(M) - k · (trace(M))²`.
- **Hough Transform**: Polar equation `ρ = x·cos(θ) + y·sin(θ)`. Collinear image points intersect at a peak accumulator vote.
- **SIFT**: Scale (DoG) + Rotation (dominant orientation) invariant -> **128-dimensional descriptor**.
- **HOG**: Gradient histograms over cells + block normalization -> Standard for **pedestrian detection**.
- **Morphology**:
  - Erosion = **Shrink** (`A ⊖ B`)
  - Dilation = **Grow** (`A ⊕ B`)
  - Opening = **Erode then Dilate** (`A ∘ B`) (removes tiny specks)
  - Closing = **Dilate then Erode** (`A • B`) (fills tiny holes)
- **Segmentation**:
  - Split & Merge = **Quadtree**
  - Watershed = **Topographic flooding** (Marker-based avoids over-segmentation)
  - Mean Shift = **Density mode seeking**
  - Normalized Cut = **Balances partition size** over simple min-cut

---

### Official MCQ Solutions (Assignment 6 / Unit 2 Questions 21–40)
| Q# | Question Summary | Correct Option | Key Rationale |
| :---: | :--- | :---: | :--- |
| **Q21** | NOT a standard step of Canny edge detector: | **D. Eigenspace projection** | Standard steps: Smoothing, Gradient, NMS, Hysteresis. |
| **Q22** | Non-maximum suppression in edge detection is used to: | **B. Thin edges to single-pixel width** | Suppresses non-peak gradient responses along normal. |
| **Q23** | Harris corner detector identifies corners based on: | **B. Eigenvalues of local structure matrix** | Eigenvalues of structure tensor M. |
| **Q24** | In Harris detector, a true corner is indicated by: | **C. Both eigenvalues large** | Large gradients along two orthogonal directions. |
| **Q25** | SIFT stands for: | **A. Scale Invariant Feature Transform** | Canonical algorithm name. |
| **Q26** | SIFT descriptors specifically designed to be invariant to: | **B. Scale and rotation** | Keypoint scale-space & orientation assignment. |
| **Q27** | HOG stands for: | **B. Histogram of Oriented Gradients** | Dense feature extraction technique. |
| **Q28** | HOG features are classically well known for: | **B. Pedestrian/human detection** | Benchmark feature for Dalal & Triggs pedestrian detector. |
| **Q29** | Morphological erosion applied to a binary object generally: | **B. Shrinks the object** | Strips away boundary pixels where SE doesn't fully fit. |
| **Q30** | Morphological dilation applied to a binary object generally: | **B. Grows the object** | Adds boundary pixels where SE touches foreground. |
| **Q31** | Morphological opening is defined as: | **B. Erosion followed by dilation** | Smooths outer contours, eliminates small noise. |
| **Q32** | Morphological closing is defined as: | **A. Dilation followed by erosion** | Bridges small breaks and fills internal holes. |
| **Q33** | Watershed segmentation treats a grayscale image as: | **B. A topographic surface to be flooded** | Catchment basins fill from local minima. |
| **Q34** | Marker-based watershed is primarily used to reduce: | **B. Over-segmentation** | Restricts flooding origins to genuine object markers. |
| **Q35** | Region growing begins from: | **B. Seed points** | Initial anchor pixels that grow into homogeneous regions. |
| **Q36** | Region splitting & merging commonly implemented using: | **B. Quad-tree structure** | 4-way recursive spatial partitioning. |
| **Q37** | In graph-based segmentation, pixels/regions represented as: | **B. Nodes of a graph** | Pixels are vertices, similarities are edge weights. |
| **Q38** | Normalized Cut criterion is designed to: | **B. Balance segment size while minimizing cut** | Avoids bias toward tiny isolated component cuts. |
| **Q39** | Mean-shift segmentation clusters pixels by finding: | **B. Modes (peaks) of local density function** | Hill-climbing along mean-shift vectors. |
| **Q40** | Active contour ("snake") evolves over iterations to: | **B. Minimize an energy function** | Balances internal elasticity with external edge attraction. |
