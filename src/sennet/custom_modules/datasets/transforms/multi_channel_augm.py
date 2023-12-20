import albumentations as A
import numpy as np


class MultiChannelAugmentation:
    
    def __init__(self,
                 random_3d_rotate: bool = False,
                 ) -> None:
        
        
        self.random_3d_rotate = random_3d_rotate
        self.random_3d_rotate_transform = A.Compose([
            A.Rotate(limit=90, p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
        ])
        
    def transform(self, data):
        
        img = data["img"]
        gt_seg_map = data["gt_seg_map"]
        
        if self.random_3d_rotate:
            
            out = self.random_3d_rotate_transform(image=img, mask=gt_seg_map)
            img_augmented, gt_seg_map_augmented = out["image"], out["mask"]
            # flip x and y axes and do it again
            img_augmented = np.transpose(img_augmented, (1, 0, 2))
            gt_seg_map_augmented = np.transpose(gt_seg_map_augmented, (1, 0, 2))
            out = self.random_3d_rotate_transform(image=img_augmented, mask=gt_seg_map_augmented)
            img_augmented, gt_seg_map_augmented = out["image"], out["mask"]
        
            img = img_augmented
            gt_seg_map = gt_seg_map_augmented
        
        
        
        
        data["img"] = img
        data["gt_seg_map"] = gt_seg_map
        
        return data
            
        