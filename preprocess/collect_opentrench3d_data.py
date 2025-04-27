# Used to convert from .ply to .npy

import os
import sys
import argparse
from plyfile import PlyData
import numpy as np

def ply_to_numpy(ply_path, output_npy_path):
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']

    x = vertex['x']
    y = vertex['y']
    z = vertex['z']
    r = vertex['red']
    g = vertex['green']
    b = vertex['blue']
    label = vertex['class']  # This works when the label is called 'class'. Look at the .ply header and check that they define it as 'class'

    data = np.stack([x, y, z, r, g, b, label], axis=-1).astype(np.float64)

    # Save as .npy
    np.save(output_npy_path, data)
    
    print(f"Converted {ply_path} -> {output_npy_path}")
    print(f"\tShape={data.shape} \n\tDtype={data.dtype} \n\tLabels={np.unique(data[:,-1])} \n")


def main():
    parser = argparse.ArgumentParser(description="Convert .ply files in a directory to .npy files.")
    parser.add_argument(
        "--data_path", 
        required=True, 
        help="Directory containing input .ply files"
    )
    parser.add_argument(
        "--save_path", 
        required=True, 
        help="Directory to save output .npy files"
    )
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    ply_files = [f for f in os.listdir(args.data_path) if f.endswith('.ply')]

    if not ply_files:
        print("No .ply files found in the specified directory.")
        return

    for ply_file in ply_files:
        input_ply = os.path.join(args.data_path, ply_file)
        output_npy = os.path.join(args.save_path, os.path.splitext(ply_file)[0] + '.npy')
        ply_to_numpy(input_ply, output_npy)


if __name__ == "__main__":
    main()

