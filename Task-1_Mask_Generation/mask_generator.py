import os
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# Cityscapes class to training ID mapping (19 classes)
CITYSCAPES_TRAIN_IDS = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4,
    17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
    23: 10, 24: 11, 25: 12, 26: 13,
    27: 14, 28: 15, 31: 16, 32: 17, 33: 18
}

def convert_mask(gt_path, mode='multiclass'):
    """Converts the Cityscapes color or ID masks to a multi-class or binary mask."""
    mask = np.array(Image.open(gt_path))
    
    if mode == 'binary':
        binary_mask = np.isin(mask, list(CITYSCAPES_TRAIN_IDS.keys())).astype(np.uint8)
        return binary_mask * 255  # 0 or 255
    elif mode == 'multiclass':
        out_mask = np.full(mask.shape, 255, dtype=np.uint8)  # 255 = ignore index
        for k, v in CITYSCAPES_TRAIN_IDS.items():
            out_mask[mask == k] = v
        return out_mask
    else:
        raise ValueError("Mode must be 'binary' or 'multiclass'")

def process_cityscapes(image_dir, label_dir, output_mask_dir, mode='multiclass'):
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    output_mask_dir = Path(output_mask_dir)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    for city_folder in tqdm(sorted(label_dir.glob('*'))):
        for gt_file in sorted(city_folder.glob('*_gtFine_labelIds.png')):
            try:
                relative_path = gt_file.relative_to(label_dir)
                mask = convert_mask(gt_file, mode=mode)

                # Save mask as PNG
                out_path = output_mask_dir / relative_path
                out_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(mask).save(out_path)
            except Exception as e:
                print(f"Skipping {gt_file}: {e}")

if __name__ == "__main__":
    process_cityscapes(
        image_dir="Cityscapes/leftImg8bit/train",
        label_dir="Cityscapes/gtFine/train",
        output_mask_dir="Cityscapes/train_masks_multiclass",
        mode='multiclass'  # change to 'binary' if needed
    )
