import argparse
import os
import numpy as np
from plyfile import PlyData


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess Toronto-3D dataset for few-shot learning")
    parser.add_argument('--data_path', type=str, required=True, help='Path to raw .ply files')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save .npy files')
    return parser.parse_args()


def process_ply_file(ply_path):
    ply_data = PlyData.read(ply_path)
    vertex = ply_data['vertex']

    xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)
    rgb = np.stack([vertex['red'], vertex['green'], vertex['blue']], axis=-1)
    labels = np.array(vertex['scalar_Label'])  # ✅ semantic labels are stored here

    data = np.concatenate([xyz, rgb, labels[:, np.newaxis]], axis=1)  # shape: N x 7
    return data


def main():
    args = parse_args()
    save_data_path = os.path.join(args.save_path, 'data')
    os.makedirs(save_data_path, exist_ok=True)

    print("✅ Using 'scalar_Label' from .ply files directly for semantic labels")

    # Process all .ply files in the directory
    for fname in os.listdir(args.data_path):
        if fname.endswith('.ply'):
            ply_path = os.path.join(args.data_path, fname)
            data = process_ply_file(ply_path)

            room_name = os.path.splitext(fname)[0]
            out_path = os.path.join(save_data_path, f'{room_name}.npy')
            np.save(out_path, data)
            print(f"✅ Saved {out_path}, shape: {data.shape}, labels: {np.unique(data[:, 6])}")

    print("🎉 All scans processed successfully.")


if __name__ == '__main__':
    main()
