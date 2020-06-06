import os
import json
import numpy as np
import torch
from PIL import Image

def convert_bbox_voc_to_coco(bbox):
    x,y,w,h = bbox
    x0 = x - w/2
    y0 = y - h/2
    x1 = x + w/2
    y1 = y + h/2
    return (x0,y0,x1,y1)

class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, annotation_file, transforms=None):
        with open(annotation_file) as f:
            self.annotations = json.load(f)

        self.transforms = transforms

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = 'data/' + self.annotations[idx]['img_path'] + '.jpg'
        img = np.asarray(Image.open(img_path).convert("RGB"))

        boxes = self.annotations[idx]['bbox']
        # boxes = [convert_bbox_voc_to_coco(x) for x in boxes]

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((len(boxes),), dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        if self.transforms is not None:
            sample = {
                'image': img,
                'bboxes': boxes,
                'labels': labels
            }
            sample = self.transforms(**sample)
            img = sample['image']
            boxes = sample['bboxes']
            labels = sample['labels']
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return img, target

