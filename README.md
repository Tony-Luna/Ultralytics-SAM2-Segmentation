# Interactive Object Segmentation Tool with Ultralytics SAM2

![Demo](assets/demo.gif)

## Overview

This **Interactive Object Segmentation Tool** provides a straightforward, **local** solution for high-accuracy object segmentation using **Meta’s Segment Anything Model (SAM2)**—integrated via **Ultralytics**. It enables you to refine masks via positive (green) and negative (red) prompts directly on your images, all running on your own machine.

**Key Features**:
- **Point-Based GUI**:  
  - **Positive Prompts** (green) to include objects or regions.  
  - **Negative Prompts** (red) to remove unwanted areas (currently produces inconsistent results).
- **Simple Controls**: “Clear Last” or “Clear All” to remove points, “OK” to save segmented outputs.
- **Local Execution**: Your data stays private, with no external uploads.
- **Directory Navigation**: Easily browse multiple images in a folder.
- **SAM2 Integration**: Built around [Ultralytics SAM2](https://docs.ultralytics.com/models/sam-2/).

## Table of Contents

1. [Quick Start](#quick-start)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Model Details](#model-details)
5. [Usage Examples](#usage-examples)
6. [Configuration](#configuration)
7. [Advanced Tips](#advanced-tips)
8. [Contributing](#contributing)
9. [Future Improvements](#future-improvements)
10. [License](#license)
11. [Contact](#contact)

---

## Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Tony-Luna/Ultralytics-SAM2-Segmentation.git
   cd Ultralytics-SAM2-Segmentation
   ```

2. **Create & activate a Conda environment**:
   ```bash
   conda create -n seg_env python=3.12
   conda activate seg_env
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   > **Note**: Ensure `numpy<2` is specified in the `requirements.txt` due to known compatibility issues.

4. **Install PyTorch** (CPU or GPU):
   - **CUDA 11.8**:
     ```bash
     pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
     ```
   - **CUDA 12.1**:
     ```bash
     pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
     ```
   - **CPU Only**:
     ```bash
     pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu
     ```

5. **Adjust `config.yaml`** to suit your environment (see [Configuration](#configuration)).

6. **Run**:
   ```bash
   python main.py
   ```
   > The SAM2 model weights will auto-download if not present.

7. **Use the GUI**:
   - **Left-click** -> Positive prompt (green).
   - **Right-click** -> Negative prompt (red, experimental).
   - **Clear Last** -> Remove your most recent point.
   - **Clear All** -> Remove all points for the current image.
   - **OK** -> Save a cropped PNG of your segmentation.
   - **`<` / `>`** -> Navigate images if `input_image_path` is a directory.

---

## Requirements

- **Python** 3.12+
- **Conda** (recommended)
- **PyTorch** 2.2.2+
- **Ultralytics** (for SAM2 integration)
- **OpenCV & Tkinter** (for GUI)
- **Numpy < 2** (compatibility constraint)

All dependencies are listed in `requirements.txt`.

---

## Installation

Refer to the [Quick Start](#quick-start). In short:

1. Create a Python 3.12 environment (via Conda).  
2. `pip install -r requirements.txt`.  
3. Install PyTorch for CPU or CUDA usage.  
4. `python main.py` to launch the interactive segmentation tool.

---

## Model Details

This tool supports the following SAM2 variants from [Ultralytics documentation](https://docs.ultralytics.com/models/sam-2/). To switch models, **edit `model_path`** in `config.yaml` accordingly. The correct weights file will be automatically downloaded.

- **SAM 2 tiny**: `sam2_t.pt`
- **SAM 2 small**: `sam2_s.pt`
- **SAM 2 base**: `sam2_b.pt`
- **SAM 2 large**: `sam2_l.pt`
- **SAM 2.1 tiny**: `sam2.1_t.pt`
- **SAM 2.1 small**: `sam2.1_s.pt`
- **SAM 2.1 base**: `sam2.1_b.pt`
- **SAM 2.1 large**: `sam2.1_l.pt`

Example usage in `config.yaml`:
```yaml
usage_parameters:
  model_path: "./models/sam2.1_b.pt"
```

---

## Usage Examples

### Single Image Usage

1. **Edit `config.yaml`**:
   ```yaml
   usage_parameters:
     input_image_path: "path/to/your/image.jpg"
     output_image_path: "path/to/output/folder"
     model_path: "./models/sam2.1_b.pt"
     verbosity: 2
     device: "cuda:0"
     visualization_size: 800
   ```
2. **Run**:
   ```bash
   python main.py
   ```
   Interact with the GUI:
   - Add positive or negative points.
   - Clear points individually or entirely.
   - Save results using “OK.”

### Directory Processing

If `input_image_path` is a **directory**, navigate between images using `<` / `>`. Points are saved in `.npz`, allowing you to revisit them later.

---

## Configuration

In **`config.yaml`**, the primary parameters are:

| Parameter              | Description                                                                      | Example Value                                   |
|------------------------|----------------------------------------------------------------------------------|-------------------------------------------------|
| **input_image_path**   | Single image file or a directory path.                                           | `"C:/Users/user/Pictures/Raw"`                 |
| **output_image_path**  | Where segmented PNGs and points get saved.                                       | `"C:/Users/user/Pictures/Segmented"`           |
| **model_path**         | Path for SAM2 weights (will auto-download if missing).                           | `"./models/sam2.1_b.pt"`                       |
| **verbosity**          | Logging level (0=WARNING, 1=INFO, 2=DEBUG).                                     | `2`                                            |
| **device**             | `"cpu"` or `"cuda:0"` for GPU usage.                                             | `"cuda:0"`                                     |
| **visualization_size** | Maximum dimension for displayed images.                                          | `800`                                          |

---

## Advanced Tips

- **Scaling**: The display might be resized to `visualization_size`, but segmentation runs at the original resolution.
- **Performance**: A GPU (`device: "cuda:0"`) accelerates segmentation, especially for large images.
- **Points**: Each image’s prompt points are stored as `.npz`. Upon revisiting an image, these points are automatically loaded.
- **Supported Formats**: PNG, JPG, BMP, and TIFF.

---

## Contributing

We welcome community contributions:

- **Issues**: Report bugs or request features.
- **Pull Requests**: Fork, develop, and submit PRs.
- **Discussions**: Brainstorm or propose bigger changes.

---

## Future Improvements

1. **Enhanced Negative Prompting**: Improve consistency for removing unwanted regions.
2. **Box Prompting**: Add bounding box prompts for quicker region-of-interest marking.

---

## License

This project is licensed under [AGPL-3.0](LICENSE).

---

## Contact

**Project Maintainer**  
**Antonio Luna Macías** (Tony-Luna)  
- [LinkedIn](https://www.linkedin.com/in/antoniolunamacias/)  
- [GitHub](https://github.com/Tony-Luna)

If you have any questions or suggestions, please open an issue or reach out!