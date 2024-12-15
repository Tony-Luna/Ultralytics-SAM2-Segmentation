# Interactive Object Segmentation Tool with SAM2 (Ultralytics)

![Demo](assets/demo.gif)

## Overview

This **Interactive Object Segmentation Tool** empowers you to run advanced, high-accuracy object segmentation locally—no subscriptions, no cloud servers, and no complex editors needed. By integrating Meta’s **Segment Anything Model (SAM2)** through Ultralytics, this tool delivers cutting-edge segmentation right on your machine.

**Key Advantages**:
- **Local & Private**: Full control over your data, no external uploads.
- **Ultralytics SAM2 Integration**: Accurate segmentation powered by the latest SAM2 model (see [Ultralytics SAM2 docs](https://docs.ultralytics.com/models/sam-2/#core-components)).
- **Opencv & Tkinter**: Core libraries enabling efficient image processing and a responsive GUI.
- **Simple, Intuitive GUI**: Add or remove points interactively to refine masks.
- **No Overkill Software**: Focus on segmentation alone, without large external tools.
- **Batch Processing**: Navigate multiple images, save points, and refine masks easily.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Model](#model)
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
   git clone https://github.com/yourusername/segmentation_tool.git
   cd segmentation_tool
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
   
   **Note**: Ensure `numpy<2` is specified in the `requirements.txt` due to compatibility issues.

4. **Install PyTorch** (choose one, or see [official PyTorch instructions](https://pytorch.org/get-started/locally/) for more variations):
   - CUDA 11.8:
     ```bash
     pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
     ```
   - CUDA 12.1:
     ```bash
     pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
     ```
   - CPU Only:
     ```bash
     pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu
     ```

5. **Run the tool**:
   ```bash
   python main.py
   ```
   
   On first run, SAM2 model weights will be automatically downloaded to the path specified in `config.yaml` if not already present.

6. **Interact with the GUI**:
   - Left-click on the image to add points (green "X").
   - Right-click to remove the last point.
   - Click `OK` to save results.
   - Use `<` and `>` buttons to navigate images in a directory.

---

## Requirements

- **Python**: 3.12+
- **Conda**: Recommended for environment management
- **PyTorch**: 2.2.2
- **Numpy < 2**: Required due to compatibility issues
- **OpenCV & Tkinter**: Core libraries for image processing and GUI
- **Ultralytics**: For SAM2 model integration

All dependencies are listed in `requirements.txt`.

---

## Installation

Refer to the [Quick Start](#quick-start) section.

In summary:
- Create a Python 3.12 environment.
- Install requirements (`numpy<2` is crucial).
- Install PyTorch according to your hardware capabilities.
- Run `python main.py`, and the SAM2 weights are automatically downloaded if needed.

---

## Model

You do not need to manually place the model weights. On first run, the SAM2 weights specified in `config.yaml` will be automatically downloaded. For more info on Ultralytics’ SAM2 model usage, see the [official documentation](https://docs.ultralytics.com/models/sam-2/#core-components).

---

## Usage Examples

### Single Image Processing

1. **Configure `config.yaml`**:
   ```yaml
   usage_parameters:
     input_image_path: "path/to/your/image.jpg"
     output_image_path: "path/to/output/directory"
     model_path: "./models/sam2.1_b.pt"
     verbosity: 2
     device: "cuda:0"
     visualization_size: 800
   ```

2. **Run**:
   ```bash
   python main.py
   ```
   
   Interact with the GUI to place points. Your segmented images and points are saved in the output directory.

### Directory Processing

If `input_image_path` is a directory, use the `<` and `>` buttons to navigate through images. Each image’s points and masks are saved and can be revisited.

---

## Configuration

`config.yaml` controls paths, device, verbosity, and visualization size:

```yaml
usage_parameters:
  input_image_path: "C:/Users/user/Pictures/Raw"
  output_image_path: "C:/Users/user/Pictures/Segmented"
  model_path: "./models/sam2.1_b.pt"
  verbosity: 2
  device: "cuda:0"
  visualization_size: 800
```

**Key Parameters**:
- `input_image_path`: Single image or directory path.
- `output_image_path`: Directory for saving masks and points.
- `model_path`: Location for SAM2 weights; auto-downloaded if missing.
- `verbosity`: Logging level (0=WARNING, 1=INFO, 2=DEBUG).
- `device`: `cpu` or `cuda:0`.
- `visualization_size`: Max dimension for displayed image scaling.

---

## Advanced Tips

- **Coordinate System**: The displayed image is scaled for viewing, but all segmentation occurs at the original resolution.
- **Performance**: Use a GPU for speed (`device: "cuda:0"`), or reduce `visualization_size` to improve responsiveness.
- **Supported Formats**: Common image formats (e.g., PNG, JPG, TIFF) are supported.

---

## Contributing

Contributions are welcome!  
- **Issues**: Report bugs or feature requests.
- **Pull Requests**: Fork the repo, implement changes, and submit a PR.

We appreciate your involvement in improving this tool.

---

## Future Improvements

- **Additional Prompt Types**: Integrate polygon or box-based prompts.
- **Multiple Models**: Allow easy switching between different segmentation backends.
- **Advanced Visualization**: Add overlay enhancements or label maps.

---

## License

This project is released under the [AGPL-3.0](LICENSE).

---

## Contact

For questions, feedback, or support:

**Antonio Luna Macías**  
[LinkedIn Profile](https://www.linkedin.com/in/antoniolunamacias/)