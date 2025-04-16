import argparse
import os
import numpy as np
import xml.etree.ElementTree as ET
from plyfile import PlyData


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess Mavericks dataset for room2blocks")
    parser.add_argument('--data_path', type=str, required=True, help='Path to raw dataset')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save .npy files')
    return parser.parse_args()


def parse_colors_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    color_dict = {}
    for label in root.findall('label'):
        name = label.find('name').text
        r = int(label.find('color').get('r'))
        g = int(label.find('color').get('g'))
        b = int(label.find('color').get('b'))
        color_dict[tuple([r, g, b])] = name
    return color_dict


def load_class_mapping(class_file):
    with open(class_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    return class_to_idx


def match_color_to_label(rgb, color_to_label, label_to_idx):
    rgb_tuple = tuple(rgb)
    if rgb_tuple in color_to_label:
        label_name = color_to_label[rgb_tuple]
        return label_to_idx.get(label_name, -1)
    return -1  # Unknown


def process_ply_file(ply_path, color_to_label, label_to_idx):
    ply_data = PlyData.read(ply_path)
    vertex = ply_data['vertex']

    xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)
    rgb = np.stack([vertex['red'], vertex['green'], vertex['blue']], axis=-1)

    labels = np.array([
        match_color_to_label(c, color_to_label, label_to_idx) for c in rgb
    ])

    data = np.concatenate([xyz, rgb, labels[:, np.newaxis]], axis=1)  # N x 7
    return data


def main():
    args = parse_args()
    os.makedirs(os.path.join(args.save_path, 'data'), exist_ok=True)

    # Load color-label mapping
    color_map = parse_colors_xml(os.path.join(args.data_path, 'Colors.xml'))
    label_map = load_class_mapping(os.path.join(args.data_path, 'Mavericks_classes_9.txt'))

    # Save mappings (optional for debugging)
    np.save(os.path.join(args.save_path, 'color_label_dict.npy'), color_map)
    np.save(os.path.join(args.save_path, 'label_index_dict.npy'), label_map)

    # Process .ply files
    for fname in os.listdir(args.data_path):
        if fname.endswith('.ply'):
            ply_path = os.path.join(args.data_path, fname)
            data = process_ply_file(ply_path, color_map, label_map)

            room_name = os.path.splitext(fname)[0]
            out_path = os.path.join(args.save_path, 'data', f'{room_name}.npy')
            np.save(out_path, data)
            print(f"Saved {out_path}, shape: {data.shape}")

    print("All rooms processed.")


if __name__ == '__main__':
    main()
