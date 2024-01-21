import numpy as np
import torch


class BatchTransform:
    def __init__(
        self, 
        alpha_cutmix: float,
        alpha_mixup: float,
        cutmix_prob: float,
        mixup_prob: float,
        channel_reverse_prob: float,
    ):
        self.alpha_cutmix = alpha_cutmix
        self.alpha_mixup = alpha_mixup
        self.cutmix_prob = cutmix_prob
        self.mixup_prob = mixup_prob
        self.channel_reverse_prob = channel_reverse_prob
        
    def __call__(self, batch):
        img = batch["img"]
        mask = batch["gt_seg_map"] if "gt_seg_map" in batch else None
        
        if mask is not None:
            img, mask = self.apply_cutmix_and_mixup(img, mask)
            img, mask = self.apply_channel_reverse(img, mask)
            batch["img"] = img
            batch["gt_seg_map"] = mask
            return batch
        else:
            batch["img"] = img
            return batch
        
    def apply_channel_reverse(self, img, mask):
        # Apply Channel Reverse
        if np.random.rand() < self.channel_reverse_prob:
            # Channel Reverse
            img = img.flip(2)
            mask = mask.flip(2)
            
        return img, mask
    
    def apply_cutmix_and_mixup(self, img, mask):
        # Apply Cutmix and Mixup
        if np.random.rand() < self.cutmix_prob:
            # Cutmix
            img, mask = self.cutmix(img, mask)
        if np.random.rand() < self.mixup_prob:
            # Mixup
            img, mask = self.mixup(img, mask)
            
        return img, mask    
    
    def mixup(self, img, gt_seg_map):
        # Mixup
        # https://arxiv.org/pdf/1710.09412.pdf
        
        batch_size = img.size(0)
        idx = torch.randperm(batch_size)
        lam = np.random.beta(self.alpha_cutmix, self.alpha_cutmix)

        mixed_img = lam * img + (1 - lam) * img[idx]
        mixed_labels = lam * gt_seg_map + (1 - lam) * gt_seg_map[idx]
        
        return mixed_img, mixed_labels

    def get_rand_slice_start_end(self, size, lam):
        cut_rat = np.sqrt(1.0 - lam)
        cut_len = int(size * cut_rat)

        if size - cut_len == 0:
            start = 0
        else:
            start = np.random.randint(0, size - cut_len)
        end = start + cut_len
        
        return start, end

    def cutmix(self, img, gt_seg_map):
        # Cutmix
        # https://arxiv.org/pdf/1905.04899.pdf
        
        batch_size = img.size(0)
        idx = torch.randperm(batch_size)

        shuffled_img = img[idx]
        shuffled_labels = gt_seg_map[idx]

        lam = np.random.beta(self.alpha_mixup, self.alpha_mixup)
        start, end = self.get_rand_slice_start_end(img.size(2), lam)  # it is assumed that the image is square

        mixed_img = img.clone()  
        mixed_labels = gt_seg_map.clone()

        mixed_img[:, :, start:end, start:end] = shuffled_img[:, :, start:end, start:end]
        mixed_labels[:, :, start:end, start:end] = shuffled_labels[:, :, start:end, start:end]

        return mixed_img, mixed_labels
    
    
    