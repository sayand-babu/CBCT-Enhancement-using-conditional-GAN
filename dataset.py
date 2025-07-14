#-*- coding:utf-8 -*-
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import numpy as np
import torchio as tio
import os
from glob import glob


class NieftiPairImageGenerator(Dataset):
    def __init__(self):
        # Initialize the parent Dataset class
        super().__init__()
        pathA="../../dataset/TRAINCBCTSIMULATED"
        pathB="../../dataset/TRAINCTAlignedToCBCT/"
        extensionA="nii"
        splitterA="REC-"
        extensionB="nii"
        splitterB="volume-"
        rngThreshold=0.5
        padding=[(0,0),(0,0),(0,0),(0,0)]
        augment=False
        downscale=False
        self.pathsA = sorted(glob(os.path.join(pathA, f'*.{extensionA}')),
                             key=lambda x: int(x.split(splitterA)[1].split(f'.{extensionA}')[0]))
        self.pathsB = sorted(glob(os.path.join(pathB, f'*.{extensionB}')),
                             key=lambda x: int(x.split(splitterB)[1].split(f'.{extensionB}')[0]))
        print(f'no of cbct samples :{len(self.pathsA)}')
        print(f'no of ct samples :{len(self.pathsB)}')
        
        self.rngThreshold = rngThreshold
        self.padding = padding
        self.augment = augment
        self.downscale = downscale
        effective_p = 1.0 - self.rngThreshold
        self.geometric_transforms = tio.Compose([
            tio.RandomFlip(axes=('LR',), p=effective_p),
            tio.RandomFlip(axes=('AP',), p=effective_p), 
            tio.RandomFlip(axes=('IS',), p=effective_p), 
        ])

        self.intensity_transforms = tio.Compose([
            tio.RandomBiasField(p=effective_p),
            tio.RandomBlur(p=effective_p),
            tio.RandomNoise(p=effective_p),
            tio.RandomGamma(p=effective_p)
        ])


    def __len__(self):
        return len(self.pathsA)

    def __getitem__(self, idx):
        # loading the sample at index idx 
        cbct_path = self.pathsA[idx]
        ct_path = self.pathsB[idx]
        cbct_sitk_image = sitk.ReadImage(cbct_path)
        ct_sitk_image = sitk.ReadImage(ct_path)
        cbct_np = sitk.GetArrayFromImage(cbct_sitk_image)
        ct_np = sitk.GetArrayFromImage(ct_sitk_image)

        #  Ensure 4D (D,H,W,C) after reading for consistency with process/TorchIO l
        if cbct_np.ndim == 3:
            cbct_np = np.expand_dims(cbct_np, axis=-1)
        if ct_np.ndim == 3:
            ct_np = np.expand_dims(ct_np, axis=-1)

        # Apply core preprocessing (Padding, Normalization, Downscaling)
        cbct_processed_np = self.process(cbct_np)
        ct_processed_np = self.process(ct_np)

        # Convert to PyTorch Tensors (channels-first) and float 
        cbct_tensor = torch.from_numpy(cbct_processed_np).permute(3, 0, 1, 2).float()
        ct_tensor = torch.from_numpy(ct_processed_np).permute(3, 0, 1, 2).float()

        # Apply augmentation directly to tensors (if self.augment is True)
        if self.augment and np.random.rand() < self.rngThreshold:
            cbct_tensor, ct_tensor = self.apply_augmentation(cbct_tensor, ct_tensor)
    
        return {'input':cbct_tensor, 'target':ct_tensor}

    def process(self, image):
        image = np.pad(image, self.padding, mode='constant', constant_values=0)
        image = self.normalize(image)
        image = self.center_crop(image, size=(256,256,256))
        return image
    
    def center_crop(self, img, size=(160,160,160)):
        cd, ch, cw = size
        d, h, w, c = img.shape

        # Compute required padding for each spatial dimension
        pad_d = max(cd - d, 0)
        pad_h = max(ch - h, 0)
        pad_w = max(cw - w, 0)

        # Prepare symmetric padding: ((before, after), ...)
        pad_width = (
            (pad_d // 2, pad_d - pad_d // 2),
            (pad_h // 2, pad_h - pad_h // 2),
            (pad_w // 2, pad_w - pad_w // 2),
            (0, 0)  # No padding on channel
        )

        # Apply padding if necessary
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            img = np.pad(img, pad_width, mode='constant', constant_values=0)

        # Now crop center region of size (cd, ch, cw)
        d, h, w, c = img.shape  # update shape after padding
        start_d = (d - cd) // 2
        start_h = (h - ch) // 2
        start_w = (w - cw) // 2

        end_d = start_d + cd
        end_h = start_h + ch
        end_w = start_w + cw
        return img[start_d:end_d, start_h:end_h, start_w:end_w, :]  

    def normalize(self, data, min_std=1e-7):
        mean, std = data.mean(), data.std()
        return (data - mean) / std if std > min_std else data - mean

    def apply_augmentation(self, cbct_tensor, ct_tensor):
        subject = tio.Subject(
            cbct=tio.Image(tensor=cbct_tensor),
            ct=tio.Image(tensor=ct_tensor)
        )
        subject = self.geometric_transforms(subject)
        augmented_cbct_geo = subject['cbct'].data
        augmented_ct_geo = subject['ct'].data

        subject_cbct_only = tio.Subject(
            cbct=tio.Image(tensor=augmented_cbct_geo)
        )
        subject_cbct_only = self.intensity_transforms(subject_cbct_only)

        return subject_cbct_only['cbct'].data, augmented_ct_geo
    
    def sample_conditions(self, batch_size: int):
        indices = np.random.randint(0, len(self), size=batch_size)
        condition_tensors = []

        for idx in indices:
            cbct_path = self.pathsA[idx]
            cbct_sitk_image = sitk.ReadImage(cbct_path)
            cbct_np = sitk.GetArrayFromImage(cbct_sitk_image)

            if cbct_np.ndim == 3:
                cbct_np = np.expand_dims(cbct_np, axis=-1)

            cbct_np = self.process(cbct_np)
            cbct_tensor = torch.from_numpy(cbct_np).permute(3, 0, 1, 2).float()

            if self.augment and np.random.rand() < self.rngThreshold:
                subject_cbct = tio.Subject(cbct=tio.Image(tensor=cbct_tensor))
                subject_cbct = self.geometric_transforms(subject_cbct)
                subject_cbct = self.intensity_transforms(subject_cbct)
                cbct_tensor = subject_cbct['cbct'].data

            condition_tensors.append(cbct_tensor.unsqueeze(0))  # [1, C, D, H, W]

        return torch.cat(condition_tensors, dim=0).cuda()  # [B, C, D, H, W]
