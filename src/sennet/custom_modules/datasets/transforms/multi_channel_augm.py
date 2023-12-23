import albumentations as A
import numpy as np


class MultiChannelAugmentation:
    
    def __init__(
            self,
            random_brightness_contrast: dict[str, any],
            affine: dict[str, any],
            channel_dropout: dict[str, any],
            one_of: dict[str, any],
            pixel_dropout: dict[str, any],
            zoom_in_out: dict[str, any],
            channel_inversion: dict[str, any] | None = None,
            p_any_augm: float = 0.5,
            random_crop: bool = True,
            random_3d_rotate: bool = True,
    ) -> None:
        self.p_any_augm = p_any_augm
        self.random_crop = random_crop
        self.random_3d_rotate = random_3d_rotate
        self.random_brightness_contrast = random_brightness_contrast
        self.affine = affine
        self.channel_dropout = channel_dropout
        self.one_of = one_of
        self.pixel_dropout = pixel_dropout
        self.zoom_in_out = zoom_in_out
        
        self.random_3d_rotate_transform = A.Compose([
            A.Rotate(limit=90, p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
        ])
        
        self.img_augmentations = A.Compose([
            A.RandomBrightnessContrast(
                p=random_brightness_contrast["p"],
                brightness_limit=random_brightness_contrast["brightness_limit"],
                contrast_limit=random_brightness_contrast["contrast_limit"]
            ),
            A.Affine(
                scale=affine["scale"],
                translate_percent=affine["translate_percent"],
                p=affine["p"]
            ),
            A.ChannelDropout(
                channel_drop_range=channel_dropout["channel_drop_range"],
                p=channel_dropout["p"]
            ),
            A.OneOf([
                A.GaussianBlur(),
                A.MotionBlur(),
            ], p=one_of["p"]),
            A.PixelDropout(
                per_channel=pixel_dropout["per_channel"],
                p=self.pixel_dropout["p"]
            ),
            A.ShiftScaleRotate(
                shift_limit=(0, 0),
                scale_limit=zoom_in_out["scale_limit"],
                # scale_limit=tuple(augmentations["zoom_in_out"]["scale_limit"]),
                rotate_limit=(0,0),
                p=zoom_in_out["p"]
            )
        ])
        self.channel_inversion_params = channel_inversion
            
    def transform(self, data):
        
        if self.p_any_augm < np.random.rand():
            return data
        
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
        
        # randomly invert a min_axis
        if np.random.rand() < self.channel_inversion_params["p"]:
            img = np.flip(img, axis=2).copy()
            mask = np.flip(mask, axis=2).copy()
        
        return img, mask
