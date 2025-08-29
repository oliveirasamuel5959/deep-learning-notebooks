# Image Processing with OpenCV and Matplotlib

This Python script demonstrates **basic image processing** techniques using [OpenCV](https://opencv.org/) and [Matplotlib](https://matplotlib.org/).  
It reads an image, converts it to grayscale, applies binary thresholding, and displays the original, grayscale, and binary versions side-by-side.

---

## 📦 Requirements

Make sure you have **Python 3.7+** installed.  
Then, install the required dependencies:

```bash
pip install opencv-python matplotlib
```

---

## 📂 Project Structure

```
project/
│
├── images/
│   └── image.jpg       # Your input image (replace with your own)
├── dim-reduction.py           # The provided Python code
└── README.md           # This file
```

---

## ▶️ How to Run

1. **Clone or download** this repository to your local machine.
2. Place an image named `image.jpg` inside the `images/` folder.
3. Run the script:

```bash
cd dim-reduction
python dim-reduction.py
```

---

## 📝 Code Explanation

### 1. **Importing Libraries**
```python
import os
import cv2
import matplotlib.pyplot as plt
```
- `cv2` – For image reading, conversion, and processing.
- `matplotlib.pyplot` – For visualizing the images.

---

### 2. **Reading the Image**
```python
image_path = "images/image.jpg"
image = cv2.imread(image_path)
```
- Loads the image from the `images` folder in **BGR** format (OpenCV default).

---

### 3. **Grayscale Conversion**
```python
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```
- Converts the original color image to **grayscale**.

---

### 4. **Binary Thresholding**
```python
thresh = 127
im_bw = cv2.threshold(grayscale_image, thresh, 255, cv2.THRESH_BINARY)[1]
```
- Converts the grayscale image into a **binary (black and white)** image.
- Pixels with values above `127` become white; others become black.

---

### 5. **Displaying the Results**
```python
fig, axs = plt.subplots(1, 3)
```
- Creates a figure with **3 subplots**:
  1. **Original Color Image**
  2. **Grayscale Image**
  3. **Binary Image**

---

## 📷 Output Example

If your input image looks like this:

*(example placeholder)*

The script will output something like:

| Color Image | Grayscale Image | Binary Image |
|-------------|-----------------|--------------|
| ![](example_color.png) | ![](example_gray.png) | ![](example_binary.png) |

---

## 💡 Notes
- You can replace `image.jpg` with any image format supported by OpenCV.
- Adjust the `thresh` value to control binary threshold sensitivity.
- This is a basic demonstration; more complex image processing techniques can be added.

---
