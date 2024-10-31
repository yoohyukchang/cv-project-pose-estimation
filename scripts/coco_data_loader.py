from pycocotools.coco import COCO
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F

# COCO API paths for train and validation datasets
TRAIN_ANNOTATION_PATH = "scripts/annotations/person_keypoints_train2017.json"
VAL_ANNOTATION_PATH = "scripts/annotations/person_keypoints_val2017.json"
TRAIN_IMAGE_PATH = "http://images.cocodataset.org/train2017/"
VAL_IMAGE_PATH = "http://images.cocodataset.org/val2017/"

class PersonKeypointsDataset(Dataset):
    def __init__(self, annotation_file, image_path, transform=None, target_transform=None):
        self.coco = COCO(annotation_file)
        self.image_path = image_path
        self.transform = transform
        self.target_transform = target_transform
        self.person_ids = self.coco.getCatIds(catNms=['person'])
        self.img_ids = self.coco.getImgIds(catIds=self.person_ids)
        
        # Build list of (image_id, annotation) tuples
        self.samples = []
        for img_id in self.img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.person_ids, iscrowd=False)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                if 'keypoints' in ann and ann['num_keypoints'] > 0:
                    self.samples.append((img_id, ann))
                    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_id, ann = self.samples[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_url = self.image_path + img_info['file_name']
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        
        # Get bounding box
        bbox = ann['bbox']  # [x, y, w, h]
        x, y, w, h = bbox
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        
        # Crop image to bounding box
        img = img.crop((x1, y1, x2, y2))
        
        orig_w, orig_h = img.size
        
        # Adjust keypoints
        keypoints = np.array(ann['keypoints']).reshape(-1, 3)
        keypoints[:, 0] -= x1
        keypoints[:, 1] -= y1
        
        # Resize image and keypoints
        new_size = (192, 256)  # (width, height)
        img = img.resize(new_size, resample=Image.BILINEAR)
        
        scale_x = new_size[0] / orig_w
        scale_y = new_size[1] / orig_h
        keypoints[:, 0] = (keypoints[:, 0] * scale_x)
        keypoints[:, 1] = (keypoints[:, 1] * scale_y)
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        else:
            img = F.to_tensor(img)
        
        # Generate heatmaps
        if self.target_transform:
            target = self.target_transform(keypoints)
        else:
            target = torch.tensor(keypoints, dtype=torch.float32)
        
        return img, target

class HeatmapGenerator:
    def __init__(self, output_size, num_keypoints, sigma=2):
        self.output_size = output_size  # (height, width)
        self.num_keypoints = num_keypoints
        self.sigma = sigma
        
    def __call__(self, keypoints):
        heatmaps = np.zeros((self.num_keypoints, self.output_size[0], self.output_size[1]), dtype=np.float32)
        tmp_size = self.sigma * 3
        
        for i in range(self.num_keypoints):
            kp = keypoints[i]
            x, y, v = kp
            if v > 0:  # Visible or occluded keypoints
                # Coordinate in output heatmap
                x = x * self.output_size[1] / 192  # Width scaling
                y = y * self.output_size[0] / 256  # Height scaling
                
                # x and y are float coordinates
                ul = [int(x - tmp_size), int(y - tmp_size)]
                br = [int(x + tmp_size + 1), int(y + tmp_size + 1)]
                
                if ul[0] >= self.output_size[1] or ul[1] >= self.output_size[0] or br[0] < 0 or br[1] < 0:
                    continue
                
                # Generate Gaussian
                size = 2 * tmp_size + 1
                x_coords = np.arange(0, size, 1, np.float32)
                y_coords = x_coords[:, np.newaxis]
                x0 = y0 = size // 2
                g = np.exp(- ((x_coords - x0) ** 2 + (y_coords - y0) ** 2) / (2 * self.sigma ** 2))
                
                # Usable Gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.output_size[1]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.output_size[0]) - ul[1]
                
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.output_size[1])
                img_y = max(0, ul[1]), min(br[1], self.output_size[0])
                
                heatmaps[i][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
                
        return torch.tensor(heatmaps, dtype=torch.float32)

def get_coco_dataloader(batch_size=32, phase="train", transform=None, target_transform=None):
    """
    Returns a DataLoader for COCO dataset.
    """
    if phase == "train":
        annotation_file = TRAIN_ANNOTATION_PATH
        image_path = TRAIN_IMAGE_PATH
    elif phase == "val":
        annotation_file = VAL_ANNOTATION_PATH
        image_path = VAL_IMAGE_PATH
    else:
        raise ValueError("Invalid phase. Choose between 'train' and 'val'.")
    
    dataset = PersonKeypointsDataset(annotation_file=annotation_file, image_path=image_path, transform=transform, target_transform=target_transform)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    return dataloader
