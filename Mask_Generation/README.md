# Cityscapes Mask Conversion

This project converts Cityscapes dataset label IDs into multiclass or binary masks for semantic segmentation tasks. It provides tools to process label images into usable masks for training and evaluation.

## Features:
- Converts Cityscapes ground truth label IDs to:
  - Multiclass masks (19 classes)
  - Binary foreground/background masks
- Includes exception handling for corrupted files
- Supports easy reproducibility via `uv` for Python environment management

## Requirements:
- Python 3.8+
- `uv` for environment management
- Dependencies: `numpy`, `pillow`, `tqdm`

## Usage:
1. Set up environment: `uv venv .venv && uv pip install -r requirements.txt`
2. Run the script: `python mask_generator.py --mode multiclass --output_dir Cityscapes/train_masks`

## License:
MIT
