import albumentations as A
import numpy as np


class MultiChannelAugmentation:
    
    def __init__(self,
                 random_3d_rotate: bool = False,
                 augmentations: any = None,
                 ) -> None:
        
        self.augmentations = augmentations
        self.random_3d_rotate = random_3d_rotate
        self.random_3d_rotate_transform = A.Compose([
            A.Rotate(limit=90, p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
        ])
        
        if augmentations is not None:
            self.img_augmentations = A.Compose([
                A.RandomBrightnessContrast(
                    p=self.augmentations["random_brightness_contrast"]["p"], 
                    brightness_limit=self.augmentations["random_brightness_contrast"]["brightness_limit"], 
                    contrast_limit=self.augmentations["random_brightness_contrast"]["contrast_limit"]
                ),
                A.Affine(
                    scale=self.augmentations["affine"]["scale"], 
                    translate_percent=self.augmentations["affine"]["translate_percent"], 
                    p=self.augmentations["affine"]["p"]
                ),
                A.ChannelDropout(
                    channel_drop_range=self.augmentations["channel_dropout"]["channel_drop_range"], 
                    p=self.augmentations["channel_dropout"]["p"]
                ),
                A.OneOf([
                    A.GaussianBlur(),
                    A.MotionBlur(),
                ], p=self.augmentations["one_of"]["p"]),
                A.PixelDropout(
                    per_channel=self.augmentations["pixel_dropout"]["per_channel"], 
                    p=self.augmentations["pixel_dropout"]["p"]
                ),
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
        
        if self.augmentations is not None:
            # apply augmentations
            out = self.img_augmentations(image=img, mask=gt_seg_map)
            img, gt_seg_map = out["image"], out["mask"]

        
        data["img"] = img
        data["gt_seg_map"] = gt_seg_map
        
        return data
            
        