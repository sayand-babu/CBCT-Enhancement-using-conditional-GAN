#-*- coding:utf-8 -*-
from torchvision.transforms import RandomCrop, Compose, ToPILImage, Resize, ToTensor, Lambda
from diffusion_model.trainer import GaussianDiffusion, Trainer
from diffusion_model.unet import create_model
from dataset import NieftiPairImageGenerator
import argparse
import torch
import os 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["CUDA_DEVICE_     # no of feature map in first layerORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBL     # no of feature map in first layerE_DEVICES"]="0"


# +

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', type=int, default=256)
    parser.add_argument('--depth_size', type=int, default=256)
    parser.add_argument('--num_channels', type=int, default=32)
    parser.add_argument('--num_res_blocks', type=int, default=1)
    parser.add_argument('--num_class_labels', type=int, default=0)
    parser.add_argument('--train_lr', type=float, default=1e-5)
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=5500)
    parser.add_argument('--timesteps', type=int, default=250)
    parser.add_argument('--save_and_sample_every', type=int, default=1000)
    parser.add_argument('--with_condition', action='store_true')  # Use this for flags
    parser.add_argument('-r', '--resume_weight', type=str, default="results/model-4.pt")
    
    args, unknown = parser.parse_known_args()  # ✅ Prevents Jupyter crash
    return args

args = get_args()
# -


# %tb

input_size = args.input_size
depth_size = args.depth_size
num_channels = args.num_channels
num_res_blocks = args.num_res_blocks
num_class_labels = args.num_class_labels
save_and_sample_every = args.save_and_sample_every
with_condition = True
resume_weight = args.resume_weight
train_lr = args.train_lr



dataset = NieftiPairImageGenerator()
print(len(dataset))

in_channels =2
out_channels = 1

model = create_model(
    input_size,
    num_channels,
    num_res_blocks,
    use_checkpoint=True,
    use_fp16=True,
    in_channels=in_channels,
    out_channels=out_channels
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = input_size,
    depth_size = depth_size,
    timesteps = args.timesteps,
    loss_type = 'l1', 
    with_condition=with_condition,
    channels=out_channels
).cuda()




trainer = Trainer(
    diffusion,
    dataset,
    step_start_ema =1000,
    update_ema_every = 10,    
    image_size = input_size,
    depth_size = depth_size,
    train_batch_size = args.batchsize,
    train_lr = train_lr,
    train_num_steps = args.epochs,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    fp16 = True,                     # turn on mixed precision training with apex
    with_condition=with_condition,
    save_and_sample_every = save_and_sample_every,
)

trainer.train()

# %tb





