from pathlib import Path
import argparse
from PIL import Image, ImageOps
from tqdm import tqdm
import cv2

import os
import pathlib

CD = pathlib.Path(__file__).parent.resolve()
os.chdir(CD)

from faceDetector import FaceFeatures

face = FaceFeatures()

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'}


def process_image(img):
    result = face.run_faceDet([img])
    if len(result[0]):
        aligned, bbox, score = result[0]
        return aligned


def process_file(src_path: Path, src_root: Path, dst_root: Path, args):
    rel = src_path.relative_to(src_root)
    dst_path = dst_root.joinpath(rel)


    dst_path.parent.mkdir(parents=True, exist_ok=True)

    try:
            
        img = cv2.imread(src_path)

        out_img = process_image(img)
        if out_img is not None:
            cv2.imwrite(dst_path, out_img)

    except Exception as e:
        print(f"Error processing {src_path}: {e}")


def find_images(root: Path):
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def parse_args():
    p = argparse.ArgumentParser(description="Batch process images (preserve structure).")
    p.add_argument('--src', required=True, help="Source folder to scan")
    p.add_argument('--dst', required=True, help="Destination folder where results will be written")
    return p.parse_args()


def main():
    args = parse_args()
    src_root = Path(args.src).resolve()
    dst_root = Path(args.dst).resolve()

    if not src_root.exists() or not src_root.is_dir():
        raise SystemExit(f"Source folder does not exist or is not a directory: {src_root}")

    images = list(find_images(src_root))
    if not images:
        print("No images found.")
        return

    iterator = tqdm(images, desc="Processing images", unit="img")
    for img_path in iterator:
        process_file(img_path, src_root, dst_root, args)

    print("Done.")


if __name__ == '__main__':
    main()
