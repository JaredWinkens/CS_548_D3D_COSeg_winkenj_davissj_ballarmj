import os
import argparse
import numpy as np
from pathlib import Path

def preprocess_with_labels(txt_file, labels_file, save_path):
    print(f"Processing: {txt_file.name}")

    data = np.loadtxt(txt_file)  # x, y, z, intensity, r, g, b
    labels = np.loadtxt(labels_file).astype(np.int32)

    if data.shape[0] != labels.shape[0]:
        raise ValueError(f"Mismatch in number of points and labels: {txt_file.name}")

    xyz = data[:, :3]
    rgb = data[:, 4:7] / 255.0  # normalize RGB
    label_col = labels[:, None]

    processed = np.concatenate([xyz, rgb, label_col], axis=1)  # shape [N, 7]

    out_file = save_path / (txt_file.stem + ".npy")
    np.save(out_file, processed)
    print(f"Saved: {out_file} ({processed.shape[0]} points)")

def main():
    parser = argparse.ArgumentParser(description="Preprocess Semantic3D .txt + .labels into .npy format")
    parser.add_argument('--data_path', type=str, required=True, help='Directory with .txt and .labels files')
    parser.add_argument('--save_path', type=str, required=True, help='Directory to save processed .npy files')
    args = parser.parse_args()

    data_path = Path(args.data_path)
    save_path = Path(args.save_path) / "data"
    save_path.mkdir(parents=True, exist_ok=True)

    txt_files = list(data_path.glob("*.txt"))

    for txt_file in txt_files:
        labels_file = txt_file.with_suffix(".labels")
        if not labels_file.exists():
            print(f"Skipping {txt_file.name} (no corresponding .labels file)")
            continue

        preprocess_with_labels(txt_file, labels_file, save_path)

if __name__ == "__main__":
    main()
