# OpenFocus User Manual

## Table of Contents

1. [Introduction](#1-introduction)
2. [Getting Started](#2-getting-started)
3. [Main Interface Overview](#3-main-interface-overview)
4. [Importing Images and Video](#4-importing-images-and-video)
5. [Viewing and Navigating Images](#5-viewing-and-navigating-images)
6. [Image Registration](#6-image-registration)
7. [Image Fusion Methods](#7-image-fusion-methods)
8. [Image Transformations](#8-image-transformations)
9. [Labels and Annotations](#9-labels-and-annotations)
10. [Settings and Configuration](#10-settings-and-configuration)
11. [Batch Processing](#11-batch-processing)
12. [Exporting Results](#12-exporting-results)
13. [Keyboard Shortcuts](#13-keyboard-shortcuts)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Introduction

OpenFocus is a professional multi-focus image fusion desktop application designed for researchers, engineers, and image processing professionals. The software processes multi-focus image sequences using advanced registration and fusion algorithms to generate all-in-focus images with full depth of field.

### Key Features

- **Multi-Focus Image Fusion**: Combine multiple images with different focus points into one fully focused image
- **Multiple Fusion Algorithms**: Choose from Guided Filter, DCT, DTCWT, GFG-FGF, and StackMFF-V4 (deep learning)
- **Image Registration**: Align misaligned image sequences using ECC or Homography methods
- **Batch Processing**: Process multiple image folders simultaneously
- **Flexible Export**: Save results as individual images, folders, or GIF animations
- **Image Transformations**: Rotate, flip, and resize image stacks
- **Label Addition**: Add text labels to registered and input stacks

### Supported Input Formats

- Image files: JPG, PNG, BMP, TIFF, and other common formats
- Video files: MP4 format (automatically decoded into image sequences)
- Folders containing numbered image sequences

### Supported Output Formats

- Individual images: JPG, PNG, BMP, TIFF
- Image sequences: Saved as folders
- Animations: GIF format

---

## 2. Getting Started

### Starting the Application

After installation, launch OpenFocus by:

1. **Windows (Executable)**: Double-click the OpenFocus shortcut or executable file
2. **Python Source**: Run `python main.py` in your activated conda environment

Upon launch, you will see the main application window with a dark theme interface.

### Quick Start Workflow

1. Import your image sequence (File → Open Folder or drag-and-drop)
2. Configure fusion settings in the right panel
3. Optionally enable registration for alignment
4. Adjust kernel size if needed
5. Click "Start Render" to begin processing
6. Export your results

---

## 3. Main Interface Overview

The OpenFocus interface consists of three main areas:

### Left Panel: Image Display Area

The left portion of the window displays your images in a split view:

- **Source Image Panel (Top)**: Shows the input image sequence
- **Result Image Panel (Bottom)**: Shows the fusion or registration results
- **Navigation Slider**: Use the slider below each panel to scroll through image frames
- **Image Info Labels**: Display current frame position and image dimensions

You can resize the panels by dragging the splitter handle between them.

### Right Panel: Control Panel

The right panel contains all configuration options organized vertically:

1. **Fusion Settings**: Select fusion algorithm and kernel size
2. **Registration Settings**: Enable/disable alignment methods
3. **Action Buttons**: Reset defaults and Start Render
4. **Source Images List**: Shows all loaded images with filenames
5. **Output List**: Shows generated results

### Menu Bar

The top menu bar provides access to all functions:

- **File**: Import/export operations
- **Edit**: Image transformations and labels
- **Batch**: Batch processing
- **Settings**: Tile, Registration, and Thread configurations
- **Help**: Environment info and contact information

---

## 4. Importing Images and Video

### Opening an Image Folder

1. Go to **File → Open Folder** or press `Ctrl+O`
2. Select a folder containing your image sequence
3. A downsample dialog will appear - choose your preferred scale factor
4. Click OK to load the images

The software will automatically:
- Detect and sort image files by filename
- Display the first image in the source panel
- Populate the source images list with all frames

### Opening Video Files

1. Go to **File → Open Video** or press `Ctrl+Shift+O`
2. Select an MP4 video file
3. The video will be automatically decoded into individual frames
4. Each frame becomes part of the image stack

### Drag and Drop

You can also drag a folder directly onto the application window to import images.

### Clearing the Stack

To remove all loaded images, go to **File → Clear Stack** or press `Ctrl+W`.

---

## 5. Viewing and Navigating Images

### Navigating the Image Stack

- **Slider**: Drag the slider below the source panel to scroll through frames
- **File List**: Click any filename in the right panel's source images list
- **Keyboard**: Use arrow keys (Left/Right) to navigate

### Zooming and Panning

- **Zoom In/Out**: Mouse wheel over the image
- **Fine Zoom**: Hold `Ctrl` + mouse wheel for smaller increments
- **Fit to Window**: Double-click the image to toggle between fit and 100% view
- **Pan**: Click and drag to move the image within the panel

### Synchronized Navigation

When you navigate in the source panel, the result panel shows the corresponding result frame if available.

---

## 6. Image Registration

Image registration corrects spatial misalignment between frames in your image stack. This is essential when images have slight shifts or perspective changes.

### Registration Methods

#### ECC (Enhanced Correlation Coefficient)

- Uses optimization to find the best alignment based on correlation
- Suitable for subtle misalignments
- Provides sub-pixel accuracy
- Slower than Homography but more precise

#### Homography

- Based on feature point matching (SIFT/ORB features)
- Handles larger geometric transformations
- Faster than ECC for initial alignment
- Good for images with significant perspective changes

### Enabling Registration

1. In the right panel, locate the **Registration** group
2. Check **Align (ECC)** to enable ECC alignment
3. Check **Align (Homography)** to enable Homography alignment
4. Both methods can be used together for better results

### When to Use Registration

- Your images have visible misalignment
- Images were captured handheld
- The scene has parallax (closer objects shift between frames)
- Results show ghosting or doubling artifacts

### Registration Settings

Access additional settings via **Settings → Registration**:
- **Downscale Width**: Control preprocessing resolution (default: 1024px)
- Lower values = faster processing, potentially lower accuracy
- Higher values = slower processing, potentially higher accuracy

---

## 7. Image Fusion Methods

OpenFocus offers five fusion algorithms. Each has different characteristics suitable for various image types.

### Guided Filter (Default)

- **Algorithm**: Uses guided filtering for edge-preserving fusion
- **Best for**: General purpose, balanced speed and quality
- **Advantages**: Good edge preservation, fast processing
- **Parameter**: Kernel size (adjustable from 1-51, odd values only)
  - Smaller kernel: More local detail, potentially more noise
  - Larger kernel: Smoother results, better noise suppression

### DCT (Discrete Cosine Transform)

- **Algorithm**: Frequency-domain fusion using DCT coefficients
- **Best for**: Texture-rich images, scientific imaging
- **Advantages**: Preserves fine textures, mathematically sound
- **Parameter**: Kernel size controls the processing window

### DTCWT (Dual-Tree Complex Wavelet Transform)

- **Algorithm**: Multi-scale wavelet domain fusion
- **Best for**: Complex scenes with multiple detail levels
- **Advantages**: Excellent multi-scale analysis, good directionality
- **Parameter**: Kernel size for filtering operations

### GFG-FGF (Generalized Four邻域 Gradient - Fast Guided Filter)

- **Algorithm**: Gradient-based fusion with fast guided filtering
- **Best for**: Images with clear focus regions
- **Advantages**: Fast, good focus region detection
- **Parameter**: Kernel size adjustment

### StackMFF-V4 (Deep Learning)

- **Algorithm**: Neural network-based fusion (requires PyTorch)
- **Best for**: Highest quality results when GPU is available
- **Advantages**: State-of-the-art fusion quality, automatic optimization
- **Requirements**: PyTorch installation, optional CUDA GPU
- **Note**: May be unavailable if PyTorch is not installed

### Selecting a Fusion Method

1. In the right panel, locate the **Fusion** group
2. Select one or more methods by checking the corresponding boxes
3. Adjust kernel size if applicable
4. Click **Start Render** to process

### Algorithm Comparison Guide

| Algorithm | Speed | Quality | Best For |
|-----------|-------|---------|----------|
| Guided Filter | Fast | Good | General use |
| DCT | Medium | Good | Textures |
| DTCWT | Medium | Very Good | Complex scenes |
| GFG-FGF | Fast | Good | Focus regions |
| StackMFF-V4 | Slow (GPU) | Excellent | Best quality |

---

## 8. Image Transformations

### Rotation

1. Go to **Edit → Rotate**
2. Choose rotation type:
   - **90° Clockwise**: Rotate all images 90 degrees right
   - **90° Counter-Clockwise**: Rotate all images 90 degrees left
   - **180°**: Flip images upside down

### Flipping

1. Go to **Edit → Flip**
2. Choose flip type:
   - **Horizontal Flip**: Mirror left-to-right
   - **Vertical Flip**: Mirror top-to-bottom

### Resizing

1. Go to **Edit → Resize**
2. A dialog will appear for resizing options
3. Enter new dimensions or scale percentage
4. All images in the stack will be resized proportionally

### Transformation Order

Transformations are applied to the entire image stack, maintaining alignment between frames.

---

## 9. Labels and Annotations

### Adding Labels

1. Ensure you have rendered results or loaded images
2. Go to **Edit → Add Label**
3. Enter your label text in the dialog
4. Labels are added to both registered stack and input stack

### Deleting Labels

- **Delete Registered Stack Labels**: Removes labels from aligned images
- **Delete Input Stack Labels**: Removes labels from original images

Access these options via **Edit → Delete Label** submenu.

---

## 10. Settings and Configuration

### Tile Settings (Memory Optimization)

For large images, OpenFocus uses tile-based processing to avoid memory issues.

Access via **Settings → Tile**:

- **Tile Block Size**: Size of each processing tile (default: 1024px)
- **Tile Overlap**: Overlap between tiles for seamless blending (default: 256px)
- **Tile Threshold**: Image size above which tiles are used (default: 2048px)

**Recommendations**:
- Smaller block size: Lower memory usage, slower processing
- Larger block size: Higher memory usage, faster processing
- Increase overlap if you see visible seams in results

### Registration Settings

Access via **Settings → Registration**:

- **Downscale Width**: Preprocessing resolution for registration
  - Default: 1024px
  - Lower for speed, higher for accuracy

### Thread Count Settings

Control CPU parallel processing:

Access via **Settings → Thread Count Settings**:

- Set the number of worker threads (default: 4)
- Match your CPU core count for optimal performance
- Higher values = faster processing, more CPU usage

---

## 11. Batch Processing

Process multiple folders automatically:

1. Go to **Batch → Batch Processing** or click the batch option
2. In the batch dialog:
   - **Add Folders**: Select folders containing image sequences
   - **Remove Folders**: Remove selected folders from the list
   - **Clear All**: Remove all folders
3. Configure output settings:
   - **Save to Subfolder**: Creates results in each input folder
   - **Save to Same Folder**: Overwrites or saves alongside input
   - **Custom Folder**: Specify output location
4. Review current fusion/registration settings
5. Click **Start Batch** to begin

The batch dialog shows real-time progress. You can cancel processing at any time.

### Batch Processing Notes

- Uses the same fusion and registration settings as manual processing
- Each folder is processed independently
- Filenames include timestamp and parameters for identification

---

## 12. Exporting Results

### Saving a Single Result

1. Select the result you want to save in the output list
2. Go to **File → Save** or press `Ctrl+S`
3. Choose format and location
4. Click Save

### Saving the Registered Stack

1. Go to **File → Save Stack → Registered Stack**
2. Choose save format:
   - **Save as Folder**: Saves each aligned frame as separate image files
   - **Save as GIF**: Creates an animated GIF of the aligned sequence

### Saving the Input Stack

1. Go to **File → Save Stack → Input Stack**
2. Choose save format:
   - **Save as Folder**: Saves original frames as separate images
   - **Save as GIF**: Creates animated GIF of input sequence

### Export Formats

- **JPG**: Compressed, good quality-to-size ratio
- **PNG**: Lossless, supports transparency
- **BMP**: Uncompressed, maximum quality
- **TIFF**: High quality, supports layers
- **GIF**: Animation format

---

## 13. Keyboard Shortcuts

### File Operations

| Shortcut | Action |
|----------|--------|
| `Ctrl+O` | Open Folder |
| `Ctrl+Shift+O` | Open Video |
| `Ctrl+S` | Save Result |
| `Ctrl+Shift+S` | Save Registered Stack |
| `Ctrl+W` | Clear Stack |
| `Ctrl+Q` | Exit |

### Navigation

| Shortcut | Action |
|----------|--------|
| `Left Arrow` | Previous frame |
| `Right Arrow` | Next frame |
| `Mouse Wheel` | Zoom in/out |
| `Ctrl + Wheel` | Fine zoom |
| `Double-click` | Toggle fit/100% view |

### General

| Shortcut | Action |
|----------|--------|
| `Space` | Pan mode toggle |
| `Delete` | Delete selected item |

---

## 14. Troubleshooting

### Common Issues

#### "No Images Loaded"

- Make sure you've imported an image folder or video first
- Check that the folder contains supported image formats

#### Fusion Results Look Blurry or Ghosted

- Enable image registration (both ECC and Homography)
- Ensure images are properly aligned before fusion
- Try a different fusion algorithm
- Increase kernel size for smoother results

#### StackMFF-V4 Unavailable

- Install PyTorch: `pip install torch torchvision`
- For GPU support, install CUDA-enabled PyTorch
- Check via **Help → Environment Info**

#### Out of Memory Errors

- Enable tile processing in **Settings → Tile**
- Reduce tile block size
- Use smaller kernel sizes
- Resize images to smaller dimensions

#### Registration Fails

- Try only one registration method at a time
- Reduce the downscale width in **Settings → Registration**
- Ensure images have sufficient contrast and features

#### Slow Processing

- Increase thread count in **Settings → Thread Count Settings**
- Use a more powerful CPU
- Reduce image resolution
- Use simpler fusion algorithms (Guided Filter instead of StackMFF-V4)

### Getting Help

1. **Environment Info**: Go to **Help → Environment Info** to check your installation
2. **Contact Support**: Go to **Help → Contact Us** for support information

---

## Appendix: Fusion Algorithm Details

### Guided Filter

The guided filter performs edge-preserving smoothing using a guidance image. It computes local linear transformations between the guidance and input images, then applies these to produce the output. This method excels at preserving edges while reducing noise.

### DCT (Discrete Cosine Transform)

DCT transforms images to the frequency domain where fusion decisions are made based on coefficient magnitudes. High-frequency components (details) are preserved while low-frequency components (smooth regions) are merged based on local energy.

### DTCWT (Dual-Tree Complex Wavelet Transform)

DTCWT provides multi-scale, multi-directional decomposition of images. It captures features at different scales and orientations, allowing for sophisticated fusion rules that preserve both texture and structure information.

### GFG-FGF

This algorithm uses a Generalized Four-neighborhood Gaussian approach to measure local focus, combined with Fast Guided Filter for weight map refinement. It efficiently identifies and merges in-focus regions.

### StackMFF-V4

A deep learning approach using a neural network trained on large datasets of multi-focus image pairs. The network learns optimal fusion strategies automatically, providing state-of-the-art results when properly configured.

---

*OpenFocus is an open-source project. For updates, bug reports, and feature requests, please visit the project repository.*
