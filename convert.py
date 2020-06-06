import os
import math
import json

def convert_to_bbox(annotation_string):
    current_line_split = annotation_string.split()
    major_axis_radius = float(current_line_split[0])
    minor_axis_radius = float(current_line_split[1])
    angle = float(current_line_split[2])
    center_x = float(current_line_split[3])
    center_y = float(current_line_split[4])

    calc_x = math.sqrt(major_axis_radius**2 * math.cos(angle)**2 \
                        + minor_axis_radius**2 * math.sin(angle)**2)
    calc_y = math.sqrt(major_axis_radius**2 * math.sin(angle)**2 \
                        + minor_axis_radius**2 * math.cos(angle)**2)

    # bounding box
    bbox_x = center_x
    bbox_y = center_y
    bbox_w = (2 * calc_x)
    bbox_h = (2 * calc_y)
    return (bbox_x, bbox_y, bbox_w, bbox_h)


if __name__ == "__main__":
    annotation_dir = 'data/FDDB-folds'
    annotation_files = os.listdir(annotation_dir)
    annotation_files = [x for x in annotation_files if 'ellipseList' in x]

    bbox_annotation = []

    for annotation_file in annotation_files:
        with open(os.path.join(annotation_dir, annotation_file)) as annotations:
            for img_file in annotations:
                img_bbox = []
                for bbox in range(int(next(annotations))):
                    current_line = next(annotations)
                    bbox = convert_to_bbox(current_line)
                    img_bbox.append(bbox)
                bbox_annotation.append({
                    "img_path": img_file.strip(),
                    "bbox": img_bbox
                })

    with open("bbox_annotation.json", "w") as f:
        json.dump(bbox_annotation, f)