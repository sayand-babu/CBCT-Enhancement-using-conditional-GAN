#-*- coding:utf-8 -*-
from diffusion_model.trainer import GaussianDiffusion, num_to_groups
from diffusion_model.trainer import GaussianDiffusion, Trainer
from diffusion_model.unet import create_model
import nibabel as nib
import torchio as tio
import numpy as np
import argparse
import torch
import os
import glob

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# +


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputfolder', type=str, default="../../dataset/TESTCBCTSTIMULATED")
    parser.add_argument('-e', '--exportfolder', type=str, default="exports/")
    parser.add_argument('--input_size', type=int, default=256)
    parser.add_argument('--depth_size', type=int, default=256)
    parser.add_argument('--num_channels', type=int, default=32)
    parser.add_argument('--num_res_blocks', type=int, default=1)
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--num_class_labels', type=int, default=0)
    parser.add_argument('--timesteps', type=int, default=250)
    parser.add_argument('-w', '--weightfile', type=str, default="results/model-5.pt")   
    args, unknown = parser.parse_known_args()  # ✅ Prevents Jupyter crash
    return args

args = get_args()
# -

# %tb

exportfolder = args.exportfolder
inputfolder = args.inputfolder
input_size = args.input_size
depth_size = args.depth_size
batchsize = args.batchsize
weightfile = args.weightfile
num_channels = args.num_channels
num_res_blocks = args.num_res_blocks
num_samples = args.num_samples
in_channels = 2
out_channels = 1
device = "cuda"

mask_list = sorted(glob.glob(f"{inputfolder}/*.nii"))
print(len(mask_list))


def center_crop(img, size=(256, 256, 256)):
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

def normalize(data, min_std=1e-7):
    mean, std = data.mean(), data.std()
    return (data - mean) / std if std > min_std else data - mean



model = create_model(input_size, num_channels, num_res_blocks, in_channels=in_channels, out_channels=out_channels).cuda()


diffusion = GaussianDiffusion(
    model,
    image_size = input_size,
    depth_size = depth_size,
    timesteps = args.timesteps,   # number of steps
    loss_type = 'L1', 
    with_condition=True,
    channels=out_channels
).cuda()
diffusion.load_state_dict(torch.load(weightfile)['ema'])
print("Model Loaded!")

# +
img_dir = exportfolder + "/enhanced"   
msk_dir = exportfolder + "/cbct"   
os.makedirs(img_dir, exist_ok=True)
os.makedirs(msk_dir, exist_ok=True)

for k, inputfile in enumerate(mask_list):
    left = len(mask_list) - (k+1)
    print("LEFT: ", left)
    ref = nib.load(inputfile)                           #  load the data from the path 
    name = inputfile.split('-')[-1].split('.')[0]       #  take the index  
    refImg = ref.get_fdata()                            #  convert the data into numpy  
    img = np.expand_dims(refImg, axis=-1)               # increase the dimnesion to 4D
    img = normalize(img)                                # normalize the data
    img = center_crop(img, (256, 256, 256))             # apply center crop  
    img = torch.from_numpy(img).permute(3, 0, 1, 2).float()
    img= img.unsqueeze(0)
    batches = num_to_groups(num_samples, batchsize)
    steps = len(batches)
    sample_count = 0
    
    print(f"All Step: {steps}")
    counter = 0
    
    for i, bsize in enumerate(batches):
        print(f"Step [{i+1}/{steps}]")
        condition_tensors, counted_samples = [], []
        for b in range(bsize):
            condition_tensors.append(img)
            counted_samples.append(sample_count)
            sample_count += 1

        condition_tensors = torch.cat(condition_tensors, 0).cuda()
        all_images_list = list(map(lambda n: diffusion.sample(batch_size=n, condition_tensors=condition_tensors), [bsize]))
        all_images = torch.cat(all_images_list, dim=0)
        sampleImages = all_images.cpu()#.numpy()
        
        for b, c in enumerate(counted_samples):
            counter = counter + 1
            sampleImage = sampleImages[b][0]
            sampleImage = sampleImage.numpy()
            nifti_img = nib.Nifti1Image(sampleImage, affine=ref.affine)
            nib.save(nifti_img, os.path.join(img_dir, f'{name}_{counter}'))
            nib.save(ref, os.path.join(msk_dir, f'{name}_{counter}'))
        torch.cuda.empty_cache()
    print("OK!")
# -


