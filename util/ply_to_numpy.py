# Used to convert from .ply to .npy

from plyfile import PlyData
import numpy as np
import sys

def ply_to_array(ply_path, output_npy_path):
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
    
    print(f"Metadata:\n\tShape={data.shape} \n\tDtype={data.dtype} \n\tLabels={np.unique(data[:,-1])} \n")
    
    print(f"Saved to {output_npy_path}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python [this_python_script] [input_file] [output_file]")
        return
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    ply_to_array(input_path, output_path)


if __name__ == "__main__":
    main()

