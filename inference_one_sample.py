import argparse, os, sys, glob
sys.path.append(os.getcwd()+"/ldm")
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm
import numpy as np
import torch

# Initialize GPU
torch.cuda.empty_cache()  # Clear any cached memory
torch.backends.cudnn.benchmark = True  # Enable cuDNN benchmarking

# Force CUDA initialization and set default device
device = torch.device('cuda:0')
torch.cuda.set_device(device)

if torch.cuda.is_available():
    print("CUDA initialized")
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))
    
    # Force larger GPU memory allocation to ensure proper initialization
    x = torch.zeros(1000, 1000).to(device)  # Allocate ~4MB
    y = torch.zeros(1000, 1000).to(device)  # Allocate ~4MB
    z = torch.zeros(1000, 1000).to(device)  # Allocate ~4MB
    
    print("\nGPU Memory after forced allocation:")
    print("Allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
    print("Cached:", torch.cuda.memory_reserved() / 1024**2, "MB")
else:
    raise RuntimeError("CUDA is not available. Please run on a GPU-enabled machine.")

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from torch.utils.data import Dataset, DataLoader
import os, sys, yaml, pickle, shutil, tarfile, glob
import cv2
import albumentations
import numpy as np
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from functools import partial
from PIL import Image
from tqdm import tqdm
import os
import torch
from torch.utils.data import Dataset, Subset
import random
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T
transform_PIL = T.ToPILImage()
from einops import rearrange, repeat
from torchvision.utils import make_grid
from math import sqrt
from argparse import ArgumentParser

import uuid

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

##create model
def create_model(device, yaml_path, model_path):
    #load config and checkpoint
    config = OmegaConf.load(yaml_path)
    config.model['params']['ckpt_path']=model_path
    model = instantiate_from_config(config.model)
    
    # Move model to GPU first
    model = model.to(device)
    
    # Initialize DDIM sampler - no need to move to device as it's not a PyTorch module
    sampler = DDIMSampler(model)
    
    # Move model components to GPU and ensure parameters are on GPU
    if hasattr(model, 'cond_stage_model'):
        model.cond_stage_model = model.cond_stage_model.to(device)
        for param in model.cond_stage_model.parameters():
            param.data = param.data.to(device)
    
    if hasattr(model, 'first_stage_model'):
        model.first_stage_model = model.first_stage_model.to(device)
        for param in model.first_stage_model.parameters():
            param.data = param.data.to(device)
    
    if hasattr(model, 'model'):
        model.model = model.model.to(device)
        for param in model.model.parameters():
            param.data = param.data.to(device)
    
    print("\nGPU Memory after model initialization:")
    print("Allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
    print("Cached:", torch.cuda.memory_reserved() / 1024**2, "MB")
    
    return model, sampler

def process_data(image_pth, mask_pth, kernel_size=2):
    # Ensure we're using the correct device
    device = torch.device('cuda:0')
    
    mask = cv2.imread(mask_pth, cv2.IMREAD_GRAYSCALE)
    original_size = mask.shape
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    dilated_mask = Image.fromarray(dilated_mask) 
    dilated_mask = np.expand_dims(dilated_mask, axis=2)
    dilated_mask = dilated_mask.astype(np.float32) / 255.0
    dilated_mask[dilated_mask < 0.1] = 0
    dilated_mask[dilated_mask >= 0.1] = 1
    dilated_mask = dilated_mask[None].transpose(0,3,1,2)
    dilated_mask = torch.from_numpy(dilated_mask).to(device)

    image = np.array(Image.open(image_pth).convert("RGB").resize((512,512)))
    image = image.astype(np.float32) / 255.0
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image).to(device)

    mask = Image.open(mask_pth).convert("L").resize((512,512))    
    mask = np.expand_dims(mask, axis=2)
    mask = mask.astype(np.float32) / 255.0
    mask[mask < 0.1] = 0
    mask[mask >= 0.1] = 1
    mask = mask[None].transpose(0,3,1,2)
    mask = torch.from_numpy(mask).to(device)
    masked_image = (1 - mask) * image

    batch = {"image": image, "mask": dilated_mask, "masked_image": masked_image}
    
    # Move batch tensors to GPU and normalize
    for k in batch:
        batch[k] = batch[k].to(device)
        batch[k] = batch[k] * 2.0 - 1.0
    
    print("\nGPU Memory after data processing:")
    print("Allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
    print("Cached:", torch.cuda.memory_reserved() / 1024**2, "MB")
    
    imagename = image_pth.split('/')[-1].split('.')[0]
    original_image = Image.open(image_pth).convert("RGB")
    original_mask = Image.open(mask_pth).convert("L")
    
    return batch, original_size, original_image, original_mask, imagename


def inference(args):
<<<<<<< HEAD
    # Force CUDA device
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    
=======
    # config
    if torch.backends.mps.is_available():
      device = torch.device("mps")  # Use Apple Metal GPU
    else:
      device = torch.device("cpu")  # Fallback to CPU

    print(f"Using device: {device}")

>>>>>>> 1e0c767c6ef0b15ced2cac706953c200841abf60
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    
    try:
        # Move model to GPU
        model, sampler = create_model(device, args.yaml_path, args.model_path)
        model.eval()
        
        # Print GPU memory usage after model loading
        print("\nGPU Memory after model loading:")
        print("Allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
        print("Cached:", torch.cuda.memory_reserved() / 1024**2, "MB")

        logpath = os.path.join(args.log_path, args.image.split('/')[-1].replace('.','_'))
        if not os.path.exists(logpath):
            os.makedirs(logpath)
            
        try:
            batch, original_size, original_image, original_mask, imagename = process_data(args.image, args.mask, args.dilate_kernel)
            
            # Print GPU memory usage after data processing
            print("\nGPU Memory after data processing:")
            print("Allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
            print("Cached:", torch.cuda.memory_reserved() / 1024**2, "MB")
            
            print("Encoding masked image...")
            c = model.cond_stage_model.encode(batch["masked_image"].to(device))
            print("Encoded shape:", c.shape)
            
            print("Interpolating mask...")
            cc = torch.nn.functional.interpolate(batch["mask"].to(device),
                                                size=c.shape[-2:])
            print("Interpolated mask shape:", cc.shape)
            
            print("Concatenating...")
            c = torch.cat((c, cc), dim=1)
            print("Concatenated shape:", c.shape)
            
            shape = (c.shape[1]-1,)+c.shape[2:]
            print("Sampling shape:", shape)
            
            c = c.expand(args.batchsize, -1,-1,-1)
            print("Expanded shape:", c.shape)
            
            cond = c.to(device)  # Ensure conditioning is on GPU
            
            print("Starting DDIM sampling...")
            try:
                # Clear memory before sampling
                torch.cuda.empty_cache()
                
                samples_ddim, _ = sampler.sample(S=args.Steps,
                                            conditioning=cond,
                                            batch_size=c.shape[0],
                                            shape=shape,
                                            verbose=False)
                
                print("DDIM sampling completed successfully")
                
                # Clear memory before decoding
                torch.cuda.empty_cache()
                
                # Process samples in smaller batches to avoid memory issues
                batch_size = 3  # Process 3 samples at a time
                all_samples = []
                
                for i in range(0, len(samples_ddim), batch_size):
                    batch_samples = samples_ddim[i:i+batch_size]
                    # decode the latent vector (output)
                    x_samples_batch = model.decode_first_stage(batch_samples)
                    # denormalize the output
                    predicted_image_clamped = torch.clamp((x_samples_batch+1.0)/2.0,
                                                min=0.0, max=1.0)
                    all_samples.append(predicted_image_clamped)
                    
                    # Clear memory after each batch
                    torch.cuda.empty_cache()
                
                # Concatenate all processed samples
                predicted_image_clamped = torch.cat(all_samples, dim=0)
                
                # Print GPU memory usage after sampling
                print("\nGPU Memory after sampling:")
                print("Allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
                print("Cached:", torch.cuda.memory_reserved() / 1024**2, "MB")
                
                all_samples= []
                base_count = len(glob.glob(os.path.join(logpath, 'sample*')))
                grid_count = len(glob.glob(os.path.join(logpath, 'grid*')))
                
                image_array = np.array(original_image)
                mask_array = np.array(original_mask)
                
                # Process images one at a time to save memory
                for sample in predicted_image_clamped:
                    output_PIL=transform_PIL(sample)
                    output_PIL = output_PIL.resize((original_size[1], original_size[0]))
                    if args.isReplace:
                        out_array = np.array(output_PIL)
                        out_array[mask_array == 0] = image_array[mask_array == 0]
                        output_PIL = Image.fromarray(out_array)
                    output_PIL.save(os.path.join(logpath, f"sample_{base_count:05}.png"))
                    base_count += 1
                    # Clear memory after each image
                    torch.cuda.empty_cache()
                
                all_samples.append(predicted_image_clamped)
                
                # additionally, save as grid
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                grid = make_grid(grid, nrow=int(sqrt(args.batchsize)))

                # to image
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                grid = Image.fromarray(grid.astype(np.uint8))
                grid.save(os.path.join(logpath, f'grid-{grid_count:04}.png'))
                grid_count += 1
                
                # Final memory cleanup
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"\nError during DDIM sampling: {str(e)}")
                print("Full error details:")
                import traceback
                traceback.print_exc()
                raise
                
        except Exception as e:
            print(f"\nError during data processing: {str(e)}")
            print("Full error details:")
            import traceback
            traceback.print_exc()
            raise
            
    except Exception as e:
        print(f"\nError during model initialization: {str(e)}")
        print("Full error details:")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument('--yaml_path', default="ldm/models/ldm/inpainting_big/config_LAKERED.yaml", help='Path to YAML ...')
    parser.add_argument('--model_path', default="ckpt/LAKERED.ckpt", help='Path to MODEL ...')
    parser.add_argument('--log_path', default="test_log", help='Result Save Path')
    parser.add_argument('--image', default="", help='Path to IMAGE ...')
    parser.add_argument('--mask', default="", help='Path to MASK ...')
    parser.add_argument('--batchsize', default=9)
    parser.add_argument('--isReplace', default=False, help='Whether to replace the foreground')
    parser.add_argument('--dilate_kernel', default=2)
    parser.add_argument('--Steps', default=50)
    
    args = parser.parse_args()
    print('Called with args:')
    print(args)
    
    inference(args)
    
    
