# Unit 1: Image Formation & Representation, Image Processing
> **Comprehensive Study Guide & Question Bank for MTE**  
> *Subject Code:* BTCS301T | *Matches Syllabus in:* [computer_vision.txt](file:///home/akapandey/Learning/computer-vision/computer_vision.txt)

---

## Table of Contents
1. [Overview of Computer Vision and Applications](#1-overview-of-computer-vision-and-applications)
2. [Image Formation & Imaging Geometry (Pinhole & Thin Lens)](#2-image-formation--imaging-geometry)
   - *Mathematical Derivation: Thin-Lens Equation*
3. [Radiometry](#3-radiometry)
4. [Digitization (Sampling & Quantization)](#4-digitization-sampling--quantization)
5. [Coordinate Frames & Geometric Transformations](#5-coordinate-frames--geometric-transformations)
   - *Homogeneous Coordinates*
   - *Hierarchy: Rigid, Similarity, Affine, Projective*
   - *Numerical 1: 2D Rotation & Translation*
6. [Pixel (Point) Transforms & Color Transforms](#6-pixel-transforms--color-transforms)
7. [Histogram Processing & Equalization](#7-histogram-processing--equalization)
   - *Numerical 2: Step-by-Step Histogram Equalization*
8. [Filtering & Spatial Convolution](#8-filtering--spatial-convolution)
   - *Numerical 3: 2D Convolution with Averaging Kernel*
   - *Low-Pass vs. High-Pass vs. Median Filtering*
9. [Fourier Transformation & Frequency-Domain Filtering](#9-fourier-transformation--frequency-domain-filtering)
   - *Convolution Theorem & Periodic Noise Removal*
10. [Quick-Review Exam Summary & MCQ Answer Keys](#10-quick-review-exam-summary--mcq-answer-keys)

---

## 1. Overview of Computer Vision and Applications

### Intuitive Explanation (Beginner Friendly)
Think of **Computer Vision (CV)** as giving eyes and a visual brain to a computer. 
- While Computer Graphics takes 3D models/data and produces a 2D picture (**Data $\to$ Image**),
- Computer Vision does the exact reverse: it takes 2D pixel grids and reconstructs what exists in the 3D world (**Image $\to$ Understanding/Data**). It is often called **inverse graphics**.

### Formal Definition
> **Computer Vision**: An interdisciplinary field of artificial intelligence and signal processing that aims to develop algorithmic techniques enabling computers to acquire, process, analyze, and understand digital images and high-dimensional visual data to produce numerical or symbolic information.

### Real-World Applications
1. **Autonomous Vehicles**: Lane detection, obstacle avoidance, pedestrian tracking.
2. **Medical Imaging**: Automated tumor segmentation in MRI/CT scans, retinal pathology screening.
3. **Biometrics & Security**: Facial recognition unlocking, automated surveillance.
4. **Industrial Automation**: Automated defect inspection on high-speed conveyor belts.

---

## 2. Image Formation & Imaging Geometry

### Intuitive Explanation
How does real-world light turn into a photograph? 
- If you just hold a piece of paper outside, light bounces from everywhere, resulting in a blurry white sheet.
- To form an image, we must restrict rays so that each point on an object maps to exactly one point on our sensor. 
- The simplest way is a tiny pinhole (**Pinhole Camera**). Because the hole is small, only one ray from each object point gets through, creating an inverted image on the back wall.
- However, tiny pinholes let very little light through. Real cameras use a glass **lens** to gather more light and focus it onto the image plane.

```
       Object (Y)                       Aperture / Pin-hole             Image Plane (-y)
          |                                      |                               |
          *--------------------------------------O-------------------------------*
          |       <---------- do ---------->     |    <------- di / f ------->   |
```

### Key Definitions
- **Pinhole Camera**: A theoretical camera model without lenses where light rays pass through a single infinitesimal aperture (pinhole) to project an inverted image on an image plane.
- **Focal Length ($f$)**: The distance between the pinhole/lens center and the image sensor plane.
- **Optical Axis**: The perpendicular line passing through the center of the lens/pinhole.

---

### 📐 Derivation: Thin-Lens Equation
*(Directly asked in Tutorial 1 & MCQ Q18)*

#### The Logic Behind It
A thin lens bends light rays according to Snell's law. By tracing two fundamental rays from the top of an object of height $H$:
1. A ray passing straight through the optical center of the lens (undeviated).
2. A ray traveling parallel to the optical axis, which refracts through the focal point $F$ at distance $f$.

#### Step-by-Step Derivation using Similar Triangles
1. From the ray passing through the lens center:
   The triangle on the object side has height $H$ and base $d_o$ (object distance).
   The triangle on the image side has inverted height $h$ and base $d_i$ (image distance).
   By similar triangles:
   $$\frac{h}{H} = \frac{d_i}{d_o} \quad \text{(Magnification Equation)}$$

2. From the ray parallel to the axis passing through focal point $F$:
   The triangle formed between the lens plane and image plane has height $H$ at the lens, height $h$ at the sensor, and base $(d_i - f)$:
   $$\frac{h}{H} = \frac{d_i - f}{f} = \frac{d_i}{f} - 1$$

3. Equating both expressions for $\frac{h}{H}$:
   $$\frac{d_i}{d_o} = \frac{d_i}{f} - 1$$

4. Divide both sides by $d_i$:
   $$\frac{1}{d_o} = \frac{1}{f} - \frac{1}{d_i} \implies \frac{1}{f} = \frac{1}{d_o} + \frac{1}{d_i}$$

> **Formula to Remember:**
> $$\mathbf{\frac{1}{f} = \frac{1}{d_o} + \frac{1}{d_i}}$$
> * Where $f$ = focal length, $d_o$ = distance of object from lens, $d_i$ = distance of image from lens.

---

## 3. Radiometry

### Intuitive Explanation
Geometry tells us **where** a point lands on the sensor; **Radiometry** tells us **how bright or colorful** that pixel will be based on light physics.

### Formal Definition
> **Radiometry**: The branch of optical science that measures physical quantities of electromagnetic radiation (including visible light) in terms of radiant energy, radiant power (flux), irradiance (power received per unit area), and radiance (power emitted per unit solid angle per unit projected area).

* **Irradiance ($E$)**: Total light power arriving at a surface per unit area.
  - **Unit**: **Watts per square meter** ($\text{W}/\text{m}^2$).
  - *Plain English*: How much total light falls onto a $1\,\text{m} \times 1\,\text{m}$ surface patch.
* **Radiance ($L$)**: Light power emitted or reflected from a surface in a specific direction.
  - **Unit**: **Watts per square meter per steradian** ($\text{W} / (\text{m}^2 \cdot \text{sr})$).
  - *Plain English Breakdown*:
    - **$\text{W}$ (Watts)**: Light energy per second (power).
    - **$\text{m}^2$ (Square Meter)**: The surface area producing/reflecting the light.
    - **$\text{sr}$ (Steradian)**: The 3D cone angle (direction) in which the light is traveling toward your camera lens.
  - *Intuition*: Radiance is what human eyes and camera sensors directly perceive as **surface brightness** when looking at an object from a specific angle.
* **Exam Relevance *(AS03 MCQ Q2, AS01 Q10a)*:** Radiometry deals with the **measurement of light energy and intensity**.

---

## 4. Digitization (Sampling & Quantization)

### Intuitive Explanation
A real-world scene is continuous in space and continuous in brightness. A computer can only store discrete numbers. Converting a continuous image into a digital matrix requires two steps:
1. **Sampling**: Dividing the 2D plane into a grid of tiny squares (pixels). Determines **Spatial Resolution** (e.g., $1920 \times 1080$).
2. **Quantization**: Rounding the continuous light brightness of each pixel into one of a finite set of discrete numbers. Determines **Grayscale/Color Resolution** (e.g., 8-bit $= 2^8 = 256$ intensity levels from 0 to 255).

```
Continuous Image  ---(Sampling in Space)--->  Pixel Grid [x, y]
                  ---(Quantization in Amplitude)---> Discrete Levels [0 .. 255]
```

### Definitions to Memorize
* **Spatial Resolution**: The number of pixels per unit length or the grid dimensions ($M \times N$). Determined strictly by the **sampling rate** *(MCQ Q4)*.
* **Grayscale Resolution (Bit Depth $k$)**: The number of gray levels $L = 2^k$ used to represent intensity values. Determined by the **quantization levels** *(MCQ Q5)*.

---

## 5. Coordinate Frames & Geometric Transformations

### Homogeneous Coordinates
*(Asked in Make up Class 1, Q1)*
* **Intuition**: In standard Cartesian coordinates $(x, y)$, translation is an addition: $(x + t_x, y + t_y)$. This cannot be written as a single matrix multiplication. By adding a dummy coordinate $w=1$, we make translations linear matrix operations!
* **Definition**: A projective coordinate system where a 2D point $(x, y)$ is represented by a 3D vector $\begin{bmatrix} x & y & 1 \end{bmatrix}^T$. Any scalar multiple $\begin{bmatrix} wx & wy & w \end{bmatrix}^T$ represents the Cartesian point $(x/w, y/w)$.
* **Why use them?**
  1. Combines rotation, scaling, and translation into a single unified matrix multiplication.
  2. Enables representation of points at infinity (directions where $w=0$).

---

### Hierarchy of 2D Transformations
*(Asked in AS01 Q2, AS13 Q2-Q4, AS03 Q6-Q7, Q19)*

| Transformation | Matrix Form (Homogeneous) | Degrees of Freedom (DOF) | What is Preserved? | What is NOT Preserved? |
| :--- | :--- | :---: | :--- | :--- |
| **Rigid (Euclidean)** | $\begin{bmatrix} \cos\theta & -\sin\theta & t_x \\ \sin\theta & \cos\theta & t_y \\ 0 & 0 & 1 \end{bmatrix}$ | **3** (1 rotation + 2 translations) | Lengths (distances), angles, areas, parallelism | Absolute position |
| **Similarity** | $\begin{bmatrix} s\cos\theta & -s\sin\theta & t_x \\ s\sin\theta & s\cos\theta & t_y \\ 0 & 0 & 1 \end{bmatrix}$ | **4** (1 scale + 1 rotation + 2 translations) | Angles, ratios of lengths, parallelism | Absolute lengths |
| **Affine** | $\begin{bmatrix} a_{11} & a_{12} & t_x \\ a_{21} & a_{22} & t_y \\ 0 & 0 & 1 \end{bmatrix}$ | **6** (4 linear + 2 translations) | **Parallelism of lines**, straight lines, ratio of areas | Angles, lengths |
| **Projective** | $\begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix}$ | **8** (up to scale) | **Straight lines**, cross-ratio | **Parallelism** (parallel lines converge to vanishing points!) |

---

### 🧮 Numerical 1: 2D Rigid Transformation (Point Rotation + Translation)
*(Direct question from Tutorial 1)*

> **Problem Statement:**  
> Given a 2D point $P = (3, 4)$, find its new coordinates after:
> 1. A rotation of $30^\circ$ counter-clockwise about the origin, followed by
> 2. A translation of $(t_x, t_y) = (2, -1)$.

#### 1. Logic Behind the Calculation
First, rotate the vector using the 2D rotation matrix. Rotation keeps the distance to the origin constant while altering orientation. Second, add the translation displacement to shift the rotated point to its final destination.

#### 2. Formulas to Remember
$$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} + \begin{bmatrix} t_x \\ t_y \end{bmatrix}$$
* Here $\theta = 30^\circ$:
  $$\cos(30^\circ) = \frac{\sqrt{3}}{2} \approx 0.8660, \quad \sin(30^\circ) = \frac{1}{2} = 0.5$$

#### 3. Step-by-Step Working
**Step A: Apply Rotation to $(3, 4)$**
$$x_{rot} = x \cos(30^\circ) - y \sin(30^\circ) = 3(0.8660) - 4(0.5) = 2.5980 - 2.0 = 0.5980$$
$$y_{rot} = x \sin(30^\circ) + y \cos(30^\circ) = 3(0.5) + 4(0.8660) = 1.5 + 3.4640 = 4.9640$$

**Step B: Apply Translation $(2, -1)$**
$$x_{new} = x_{rot} + t_x = 0.5980 + 2 = \mathbf{2.5980 \approx 2.60}$$
$$y_{new} = y_{rot} + t_y = 4.9640 + (-1) = \mathbf{3.9640 \approx 3.96}$$

**Final Answer:**
$$P' = \mathbf{(2.60, 3.96)}$$

---

## 6. Pixel Transforms & Color Transforms

### Pixel (Point) Transforms
* **Intuition**: Modifying a pixel's color purely based on its own current value, independent of its neighbors.
* **General Form**: $s = T(r)$ where $r$ is input intensity, $s$ is output intensity.
* **Examples**:
  1. **Image Negative**: $s = (L - 1) - r$. Used to inspect details in dark medical X-rays *(MCQ Q14)*.
  2. **Thresholding (Binarization)**:
     $$s = \begin{cases} 255 & \text{if } r \ge T \\ 0 & \text{if } r < T \end{cases}$$
  3. **Contrast Stretching / Brightness adjustment**: $s = a \cdot r + b$.

### Color Transforms
* **RGB**: Additive model based on primary colors (Red, Green, Blue). Highly correlated with illumination; changing brightness changes all three channels.
* **HSV (Hue, Saturation, Value)**: *(MCQ Q13)*
  - **Hue ($H$)**: Dominant wavelength (the "color" itself, $0^\circ$ to $360^\circ$).
  - **Saturation ($S$)**: Purity or vibrancy of the color ($0$ to $100\%$).
  - **Value ($V$)**: Brightness or luminous intensity.
  - *Advantage in CV:* Decouples chromatic content ($H, S$) from lighting variations ($V$).
* **Grayscale Conversion**:
  $$Y = 0.299 R + 0.587 G + 0.114 B$$
  *(Green has the highest weight because human eyes are most sensitive to green light).*

---

## 7. Histogram Processing & Equalization

### Intuitive Explanation
A histogram is a bar chart showing how many pixels exist for each brightness level (0 to 255).
- If an image has all pixels clustered between 50 and 120, the image is **low-contrast** and looks washed out, muddy, or dark gray.
- **Histogram Equalization** spreads out this crowded cluster uniformly across the entire available range $[0, 255]$, maximizing dynamic range and making hidden details visible.

---

### 🧮 Numerical 2: Histogram Equalization Algorithm
*(Directly tested in Tutorial 1 Q4, Make up Class 2 Q2, MCQ Q8)*

#### 1. Logic Behind the Calculation
We compute the **Cumulative Distribution Function (CDF)** of the image intensities. Because the CDF always increases monotonically from 0 to 1, multiplying the CDF by the maximum grayscale level $(L-1)$ gives an optimal transfer function that flattens the output histogram.

#### 2. Formula to Remember
$$s_k = T(r_k) = \text{round}\left( (L - 1) \cdot \sum_{j=0}^k p_r(r_j) \right) = \text{round}\left( (L - 1) \cdot \text{CDF}(r_k) \right)$$
* Where:
  - $L$ = total number of gray levels (e.g., $L=8$ for 3-bit, $L=256$ for 8-bit).
  - $p_r(r_j) = \frac{n_j}{N}$ is the probability of gray level $r_j$ ($n_j$ is pixel count, $N$ is total pixels).

#### 3. Step-by-Step Approach Example
Suppose we have a tiny $4 \times 4$ image ($N=16$ pixels) with 8 gray levels ($L=8$, levels $0$ to $7$):

| Gray Level $r_k$ | Pixel Count $n_k$ | Probability $p(r_k) = n_k / 16$ | Cumulative Probability ($\text{CDF}$) | Scaled $(L-1) \times \text{CDF} = 7 \times \text{CDF}$ | Rounded Output $s_k$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | 0 | 0.00 | 0.00 | 0.00 | **0** |
| **1** | 4 | 4/16 = 0.25 | 0.25 | $7 \times 0.25 = 1.75$ | **2** |
| **2** | 6 | 6/16 = 0.375 | 0.625 | $7 \times 0.625 = 4.375$ | **4** |
| **3** | 4 | 4/16 = 0.25 | 0.875 | $7 \times 0.875 = 6.125$ | **6** |
| **4** | 2 | 2/16 = 0.125 | 1.00 | $7 \times 1.00 = 7.00$ | **7** |
| **5 to 7** | 0 | 0.00 | 1.00 | 7.00 | **7** |

> **Result:** All pixels with value 1 become 2, pixels with value 2 become 4, pixels with value 3 become 6, and pixels with value 4 become 7. The narrow input band $[1, 4]$ is successfully stretched across the full spectrum $[2, 7]$!

---

## 8. Filtering & Spatial Convolution

### Intuitive Explanation
Filtering applies a small square matrix (called a **kernel** or **mask**, e.g., $3 \times 3$) over each pixel. At every position, we multiply overlapping pixel values by kernel weights and sum them up.

```
Image Patch              Kernel                Resulting Center Pixel
[10  20  30]          [1/9  1/9  1/9]
[40  50  60]    *     [1/9  1/9  1/9]   ===>   Output = Weighted Average
[70  80  90]          [1/9  1/9  1/9]
```

### 🧮 Numerical 3: 2D Convolution of a 3×3 Patch
*(Direct Question from Tutorial 1 Q6)*

> **Problem Statement:**  
> Perform 2D convolution of the $3 \times 3$ image patch below with the given $3 \times 3$ averaging kernel. Compute the output value at the center pixel:
> $$\text{Image Patch } I = \begin{bmatrix} 10 & 20 & 30 \\ 40 & 50 & 60 \\ 70 & 80 & 90 \end{bmatrix}, \quad \text{Kernel } K = \frac{1}{9} \begin{bmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix}$$

#### 1. Logic Behind the Calculation
For an averaging filter (box filter), the kernel flips $180^\circ$ (which is identical due to symmetry). We multiply each element of the image patch by its corresponding kernel weight and sum the results.

#### 2. Formula to Remember
$$\text{Output}(x, y) = \sum_{i=-1}^1 \sum_{j=-1}^1 I(x+i, y+j) \cdot K(i, j)$$

#### 3. Step-by-Step Working
1. Compute the sum of all elements in the image patch:
   $$\text{Sum} = 10 + 20 + 30 + 40 + 50 + 60 + 70 + 80 + 90 = 450$$
2. Multiply by the kernel scale factor $\frac{1}{9}$:
   $$\text{Output} = \frac{1}{9} \times 450 = \mathbf{50.0}$$

**Final Answer:** The filtered value at the center pixel is **50**.

---

### Low-Pass vs. High-Pass vs. Non-Linear Filtering

| Filter Type | Typical 3×3 Kernel | What it does | Visual Effect | Practical Use Case |
| :--- | :---: | :--- | :--- | :--- |
| **Low-Pass (Box / Averaging)** | $\frac{1}{9}\begin{bmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix}$ | Passes low spatial frequencies, blocks high frequencies | **Smooths / Blurs** | Removing random noise before edge detection *(AS03 Q9)* |
| **High-Pass (Sharpening / Laplacian)** | $\begin{bmatrix} 0 & -1 & 0 \\ -1 & 4 & -1 \\ 0 & -1 & 0 \end{bmatrix}$ | Passes high spatial frequencies, zeroes out uniform areas | **Enhances edges & fine details** *(AS03 Q10)* | Highlighting structural boundaries, medical feature enhancement |
| **Median Filter (Non-linear)** | Sorted median of neighborhood | Replaces center pixel with the median of window | Removes impulse spikes without blurring edges | **Salt-and-Pepper Noise removal** *(AS03 Q16)* |

* **Identity Kernel:** $\begin{bmatrix} 0 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{bmatrix}$ leaves the image completely unchanged *(MCQ Q17)*.

---

## 9. Fourier Transformation & Frequency-Domain Filtering

### Intuitive Explanation
- **Spatial Domain**: We look at the image as coordinates $(x, y)$ holding pixel brightness values.
- **Frequency Domain**: We treat the image as a musical orchestra composed of 2D sinusoidal waves with varying frequencies and orientations:
  - **Low frequencies**: Broad, smooth, flat regions (sky, skin, background).
  - **High frequencies**: Rapid, abrupt changes in intensity (**sharp edges**, corners, fine textures, noise).

### Mathematical Definition: 2D Discrete Fourier Transform (DFT) Pair
*(Asked in Tutorial 1 Q7, MCQ Q11)*
1. **Forward 2D DFT** (Spatial $\to$ Frequency):
   $$F(u, v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x, y) \cdot e^{-j 2\pi \left( \frac{ux}{M} + \frac{vy}{N} \right)}$$
2. **Inverse 2D DFT** (Frequency $\to$ Spatial):
   $$f(x, y) = \frac{1}{MN} \sum_{u=0}^{M-1} \sum_{v=0}^{N-1} F(u, v) \cdot e^{j 2\pi \left( \frac{ux}{M} + \frac{vy}{N} \right)}$$

### The Convolution Theorem
*(Asked in Make up Class 2 Q5, MCQ Q15)*
> **Theorem:** Spatial domain convolution is mathematically equivalent to element-wise multiplication in the frequency domain!
> $$f(x, y) * h(x, y) \iff F(u, v) \cdot H(u, v)$$
> * *Why it matters:* Convolving an image with a large $k \times k$ kernel is computationally expensive in the spatial domain ($O(N^2 k^2)$). In the frequency domain, taking the Fast Fourier Transform (FFT), multiplying, and taking the Inverse FFT runs in $O(N^2 \log N)$!

---

### Removing Periodic Noise using Frequency-Domain Filtering
*(Asked in Tutorial 1 Q8, Assignment 1 Q4, MCQ Q20)*

#### The Visual & Mathematical Problem
Periodic noise (e.g., electrical interference from power lines, scanner grid stripes) appears as repeating sinusoidal patterns across the entire image. 
- In the spatial domain, this noise contaminates every single pixel. A spatial blurring filter cannot eliminate it without blurring the actual image edges.
- In the **Fourier magnitude spectrum**, a periodic sine wave concentrates all its energy into **sharp, concentrated symmetrical spikes/dots** away from the center DC origin!

```
     Spatial Domain Contamination              Fourier Magnitude Spectrum
     ============================              ==========================
     +--------------------------+              +------------------------+
     | //////////////////////// |              |            * (Noise)   |
     | ///// [ Image ] //////// |  ---FFT--->  |         * (DC)         |
     | //////////////////////// |              |   * (Noise)            |
     +--------------------------+              +------------------------+
        (Striped Grid Artefact)                  (Symmetric Spike Dots)
```

#### The Approach: Band-Reject / Notch Filtering
1. Compute the 2D FFT of the noisy image: $F(u, v)$.
2. Shift the zero-frequency (DC) component to the center of the spectrum.
3. Locate the coordinates $(u_0, v_0)$ and $(-u_0, -v_0)$ of the bright noise spikes.
4. Construct a **Notch Reject Filter** $H(u, v)$ that has a value of $0$ around those specific spike coordinates and $1$ everywhere else.
5. Multiply: $G(u, v) = F(u, v) \cdot H(u, v)$.
6. Take the Inverse FFT. The repeating grid pattern is completely removed while keeping object edges intact!

---

## 10. Quick-Review Exam Summary & MCQ Answer Keys

### Mnemonic / Memory Cheat Sheet for MTE
- **Pinhole Camera**: Single small aperture $\to$ Inverted image $\to$ $\frac{1}{f} = \frac{1}{d_o} + \frac{1}{d_i}$.
- **Digitization**: Sampling = Spatial Resolution (pixels); Quantization = Amplitude/Gray levels (bits).
- **Affine vs. Projective**: Affine preserves **parallel lines**; Projective preserves only **straight lines** (parallel lines meet at horizon).
- **Histogram Equalization**: Uses the **CDF** to flatten the histogram and boost contrast.
- **Noise Removal**: 
  - Salt-and-Pepper $\to$ **Median filter**
  - High-frequency Gaussian noise $\to$ **Low-pass / Gaussian filter**
  - Periodic grid noise $\to$ **Frequency-domain Notch filter**
- **Fourier Duality**: Edges in space = High frequencies in Fourier. Spatial convolution = Frequency multiplication.

---

### Official MCQ Solutions (Assignment 3 / Unit 1 Questions 1–20)
| Q# | Question Summary | Correct Option | Key Rationale |
| :---: | :--- | :---: | :--- |
| **Q1** | Pinhole camera forms image via light passing through: | **A. A single small aperture** | Definition of pinhole camera model. |
| **Q2** | Radiometry deals with the measurement of: | **B. Light energy/intensity** | Radiometry measures radiant power and flux. |
| **Q3** | Digitization of analog image consists of: | **B. Sampling and quantization** | Sampling = space, Quantization = intensity. |
| **Q4** | Spatial resolution of digital image is determined by: | **B. Sampling rate** | Grid density of samples. |
| **Q5** | Number of gray levels representing pixel intensity is determined by: | **B. Quantization** | Bit depth per sample. |
| **Q6** | Transformation including only rotation and translation: | **C. Rigid transformation** | Rigid (Euclidean) preserves lengths and angles. |
| **Q7** | Key property preserved by an affine transformation: | **B. Parallelism of lines** | Affine preserves parallel lines and area ratios. |
| **Q8** | Histogram equalization is primarily used to: | **B. Improve image contrast** | Stretches the dynamic range across all bins. |
| **Q9** | Kernel typically used for image blurring: | **C. Averaging/Gaussian kernel** | Low-pass filter smooths out sharp gradients. |
| **Q10** | High-pass spatial filter primarily enhances: | **B. Edges and fine details** | Sharp transitions have high derivatives. |
| **Q11** | Fourier transform converts an image from: | **B. Spatial domain to frequency domain** | Fundamental Fourier transform property. |
| **Q12** | In frequency domain, sharp edges correspond to: | **B. High-frequency components** | Rapid spatial intensity transitions require high frequencies. |
| **Q13** | HSV color transform separates an image into: | **B. Hue, Saturation, Value** | Decouples color from brightness. |
| **Q14** | In negative transform $s = L - 1 - r$, $L$ represents: | **A. The number of gray levels** | E.g., for 8-bit, $L=256$, so $s = 255 - r$. |
| **Q15** | Spatial convolution equals what in frequency domain: | **C. Multiplication** | The Convolution Theorem. |
| **Q16** | Pixel filter commonly used for salt-and-pepper noise: | **A. Median filter** | Non-linear rank filter eliminates outlier spikes. |
| **Q17** | An identity convolution kernel applied to an image will: | **C. Leave the image unchanged** | Center is 1, all surrounding weights are 0. |
| **Q18** | Relationship between $d_o, d_i, f$ in thin-lens model derived from: | **A. Similar triangles / lens equation** | Geometric ray tracing proof. |
| **Q19** | Projective transformation does NOT necessarily preserve: | **B. Parallelism of lines** | Railroad tracks converge in perspective. |
| **Q20** | Removing high-frequency periodic noise is best done by: | **B. Frequency-domain (Fourier) filtering**| Notch filters zero out localized spectral spikes. |
