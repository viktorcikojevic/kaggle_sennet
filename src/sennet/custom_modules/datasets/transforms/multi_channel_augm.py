import albumentations as A
import numpy as np


class MultiChannelAugmentation:
    
    def __init__(self,
                 augmentations: any = None,
                 ) -> None:
        
        self.augmentations = augmentations
        self.random_3d_rotate = augmentations["random_3d_rotate"] if augmentations is not None else False
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
                )
            ])
            
            
            self.channel_inversion_params = self.augmentations["channel_inversion"] if "channel_inversion" in self.augmentations else None
            
        
        
    def transform(self, data):
        
        img = data["img"]
        gt_seg_map = data["gt_seg_map"]
        
        # Get them to (h, w, c)
        img = np.transpose(img, (1, 2, 0))
        gt_seg_map = np.transpose(gt_seg_map, (1, 2, 0))
        
        if self.random_3d_rotate:
            
            out = self.random_3d_rotate_transform(image=img, mask=gt_seg_map)
            img_augmented, gt_seg_map_augmented = out["image"], out["mask"]
            # flip x and y axes and do it again
            img_augmented = np.transpose(img_augmented, (0, 2, 1))
            gt_seg_map_augmented = np.transpose(gt_seg_map_augmented, (0, 2, 1))
            out = self.random_3d_rotate_transform(image=img_augmented, mask=gt_seg_map_augmented)
            img_augmented, gt_seg_map_augmented = out["image"], out["mask"]
            # get back to (h, w, c)
            img_augmented = np.transpose(img_augmented, (0, 2, 1))
            gt_seg_map_augmented = np.transpose(gt_seg_map_augmented, (0, 2, 1))
        
            img = img_augmented
            gt_seg_map = gt_seg_map_augmented
            
        
        if self.augmentations is not None:
            # apply augmentations
            out = self.img_augmentations(image=img, mask=gt_seg_map)
            img, gt_seg_map = out["image"], out["mask"]
            
            # perform channel inversion
            if self.channel_inversion_params is not None:
                img, gt_seg_map = self.channel_inversion(img, gt_seg_map)

            
            
        # Get images back to (c, h, w)
        img = np.transpose(img, (2, 0, 1))
        gt_seg_map = np.transpose(gt_seg_map, (2, 0, 1))
        
        
        data["img"] = img
        data["gt_seg_map"] = gt_seg_map
        
        return data

        
    def channel_inversion(self, img, mask):   
        shape = img.shape
        # find the axis with the smallest size
        min_axis = np.argmin(shape)
        
        # randomly invert a min_axis
        if np.random.rand() < self.channel_inversion_params["p"]:
            img = np.flip(img, axis=min_axis).copy()
            mask = np.flip(mask, axis=min_axis).copy()
        
        return img, mask

        
    