# Computer Vision (BTCS301T) 👁️

[![Subject Code](https://img.shields.io/badge/Subject%20Code-BTCS301T-blue.svg)](#)
[![Status](https://img.shields.io/badge/Status-MTE%20Ready-success.svg)](#)
[![Topics](https://img.shields.io/badge/Units-5%20Units%20Structured-orange.svg)](#)

A comprehensive repository for the **Computer Vision** course, containing organized assignments, tutorials, question banks, and in-depth, beginner-friendly study guides with step-by-step mathematical derivations and numerical solutions for Mid-Term Exam (MTE) preparation.

---

## 📚 Syllabus & Course Overview

Full course syllabus and contact hours reference: [computer_vision.txt](computer_vision.txt)

| Unit | Title | Contact Hours | Status | Key Topics |
| :---: | :--- | :---: | :---: | :--- |
| **Unit 1** | [Image Formation & Processing](assignments/Unit%201/README.md) | 9 | 📘 Study Guide Ready | Pinhole & Thin-Lens geometry, Radiometry, Digitization, Affine/Rigid transforms, Histogram Equalization, 2D Convolution, Fourier Filtering. |
| **Unit 2** | [Feature Detection & Segmentation](assignments/Unit%202/README.md) | 9 | 📘 Study Guide Ready | Canny Edge Detection, Harris Corner Detector, SIFT & HOG Descriptors, Morphology (Erosion/Dilation), Watershed, Split & Merge, Normalized Cut. |
| **Unit 3** | [Camera Calibration](assignments/Unit%203/README.md) | 8 | 📘 Study Guide Ready | Intrinsic ($K$) & Extrinsic ($[R \mid t]$) parameters, Projection Matrix $P$, Radial Distortion (Barrel/Pincushion), Zhang's Method, Reprojection Error. |
| **Unit 4** | Motion Representation | 8 | 📂 Assignments Sorted | Motion field, Motion parallax, Optical flow, Brightness constancy equation, Lucas-Kanade vs. Farneback, Affine flow. |
| **Unit 5** | Motion Tracking & Recognition | 8 | 📂 Assignments Sorted | Statistical filtering, Kalman Filter, Observability, Shape alignment, Appearance-based methods, Eigenfaces/PCA. |

---

## 🎯 MTE Fast-Track Study Guides

Comprehensive markdown guides tailored for students preparing for examinations. Each topic covers:
1. **Plain-English Intuition**: Simplified explanations without assuming prior advanced math background.
2. **Formal Definitions**: Exact academic terminology for exams.
3. **Numericals & Derivations Explained**:
   - *Logic Behind the Math*
   - *Formulas to Remember*
   - *Step-by-Step Worked Solutions*
4. **Assignment Mapping**: Direct solutions and references to all assigned tutorial questions and 60+ MCQ question banks.

- 📖 **[Unit 1 Study Guide](assignments/Unit%201/README.md)**: Thin lens derivation, 2D rotation & translation numerical, CDF histogram equalization algorithm, 2D convolution with averaging kernel, periodic noise filtering.
- 📖 **[Unit 2 Study Guide](assignments/Unit%202/README.md)**: 4-step Canny algorithm, Harris structure tensor & response formula, SIFT 128-d descriptor, HOG pedestrian pipeline, morphological matrix erosion, Normalized Cut ($Ncut$).
- 📖 **[Unit 3 Study Guide](assignments/Unit%203/README.md)**: Camera projection equation ($3 \times 4$ matrix $P$), Intrinsic parameters ($f_x, f_y, c_x, c_y, s$), Barrel vs. Pincushion distortion, Zhang's checkerboard calibration, RQ decomposition, Reprojection error.

---

## 📁 Repository Structure

```text
computer-vision/
├── README.md                      # Repository overview & quick navigation
├── computer_vision.txt            # Official course syllabus
└── assignments/
    ├── README.md                  # Detailed assignment directory guide
    ├── Unit 1/                    # Image Formation & Representation, Image Processing
    │   ├── README.md              # In-depth Unit 1 Study Guide & Solved Questions
    │   ├── BTCS301T-AS01.docx     # Tutorial 1
    │   ├── BTCS301T-AS02.pdf      # Assignment 1 (Code & Experiments)
    │   ├── BTCS301T-AS03.pdf      # Unit 1 - 20 MCQs Question Bank
    │   ├── BTCS301T-AS13.pdf      # Make-up Class 1 Assignment
    │   └── BTCS301T-AS17.pdf      # Make-up Class 2 Assignment
    ├── Unit 2/                    # Feature Detection & Segmentation
    │   ├── README.md              # In-depth Unit 2 Study Guide & Solved Questions
    │   ├── BTCS301T-AS04.pdf      # Assignment 2 (Code & Experiments)
    │   ├── BTCS301T-AS05.docx     # Tutorial 2
    │   ├── BTCS301T-AS06.pdf      # Unit 2 - 20 MCQs Question Bank
    │   ├── BTCS301T-AS18.pdf      # Make-up Class 3 Assignment
    │   └── BTCS301T-AS19.pdf      # Make-up Class 4 Assignment
    ├── Unit 3/                    # Camera Calibration
    │   ├── README.md              # In-depth Unit 3 Study Guide & Solved Questions
    │   ├── BTCS301T-AS07.pdf      # Assignment 3 (Code & Experiments)
    │   ├── BTCS301T-AS08.docx     # Tutorial 3
    │   ├── BTCS301T-AS09.pdf      # Unit 3 - 20 MCQs Question Bank
    │   └── BTCS301T-AS20.pdf      # Make-up Class 5 Assignment
    ├── Unit 4/                    # Motion Representation
    │   ├── BTCS301T-AS10.pdf      # Assignment 4 (Optical Flow & Video)
    │   ├── BTCS301T-AS11.docx     # Tutorial 4
    │   └── BTCS301T-AS12.pdf      # Unit 4 - 20 MCQs Question Bank
    └── Unit 5/                    # Motion Tracking & Object Recognition
        ├── BTCS301T-AS14.pdf      # Assignment 5 (Kalman Filter & Eigenfaces)
        ├── BTCS301T-AS15.docx     # Tutorial 5
        └── BTCS301T-AS16.pdf      # Unit 5 - 20 MCQs Question Bank
```

---

## 👨‍💻 Author
- **GitHub**: [@AkaPranav](https://github.com/AkaPranav)
- **Course**: Computer Vision (BTCS301T)
