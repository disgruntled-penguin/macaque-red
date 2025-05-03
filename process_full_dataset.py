import os
import argparse
import glob
from tqdm import tqdm
import torch
from inference_one_sample import create_model, process_data
import numpy as np
from PIL import Image
from torchvision.utils import make_grid
from einops import rearrange
from math import sqrt

# Set memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def process_dataset(args):
    # Initialize device
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    train_output_dir = os.path.join(args.output_dir, 'train')
    val_output_dir = os.path.join(args.output_dir, 'validation')
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(val_output_dir, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model, sampler = create_model(device, args.yaml_path, args.model_path)
    model.eval()
    
    # Process training set
    print("\nProcessing training set...")
    process_split(model, sampler, 
                 os.path.join(args.dataset_dir, 'train/images'),
                 os.path.join(args.dataset_dir, 'train/masks'),
                 train_output_dir,
                 args)
    
    # Process validation set
    print("\nProcessing validation set...")
    process_split(model, sampler,
                 os.path.join(args.dataset_dir, 'validation/images'),
                 os.path.join(args.dataset_dir, 'validation/masks'),
                 val_output_dir,
                 args)

def process_split(model, sampler, image_dir, mask_dir, output_dir, args):
    # Get all image files
    image_files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
    mask_files = sorted(glob.glob(os.path.join(mask_dir, '*.png')))
    
    # Ensure we have matching pairs
    assert len(image_files) == len(mask_files), "Number of images and masks don't match"
    
    # Process each pair
    for img_path, mask_path in tqdm(zip(image_files, mask_files), total=len(image_files)):
        try:
            # Create output directory for this sample
            sample_name = os.path.splitext(os.path.basename(img_path))[0]
            sample_output_dir = os.path.join(output_dir, sample_name)
            os.makedirs(sample_output_dir, exist_ok=True)
            
            # Process the sample
            batch, original_size, original_image, original_mask, _ = process_data(img_path, mask_path, args.dilate_kernel)
            
            # Encode and sample
            c = model.cond_stage_model.encode(batch["masked_image"].to(device))
            cc = torch.nn.functional.interpolate(batch["mask"].to(device), size=c.shape[-2:])
            c = torch.cat((c, cc), dim=1)
            shape = (c.shape[1]-1,)+c.shape[2:]
            c = c.expand(args.batchsize, -1,-1,-1)
            cond = c.to(device)
            
            # Clear memory before sampling
            torch.cuda.empty_cache()
            
            # Sample
            samples_ddim, _ = sampler.sample(S=args.Steps,
                                          conditioning=cond,
                                          batch_size=c.shape[0],
                                          shape=shape,
                                          verbose=False)
            
            # Process samples in batches to save memory
            batch_size = 3
            all_samples = []
            
            for i in range(0, len(samples_ddim), batch_size):
                batch_samples = samples_ddim[i:i+batch_size]
                x_samples_batch = model.decode_first_stage(batch_samples)
                predicted_image_clamped = torch.clamp((x_samples_batch+1.0)/2.0,
                                            min=0.0, max=1.0)
                all_samples.append(predicted_image_clamped)
                torch.cuda.empty_cache()
            
            # Concatenate all processed samples
            predicted_image_clamped = torch.cat(all_samples, dim=0)
            
            # Save individual samples
            image_array = np.array(original_image)
            mask_array = np.array(original_mask)
            
            for i, sample in enumerate(predicted_image_clamped):
                output_PIL = Image.fromarray((sample.cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8))
                output_PIL = output_PIL.resize((original_size[1], original_size[0]))
                
                if args.isReplace:
                    out_array = np.array(output_PIL)
                    out_array[mask_array == 0] = image_array[mask_array == 0]
                    output_PIL = Image.fromarray(out_array)
                
                output_PIL.save(os.path.join(sample_output_dir, f"sample_{i:05}.png"))
                torch.cuda.empty_cache()
            
            # Save grid
            grid = torch.stack([predicted_image_clamped], 0)
            grid = rearrange(grid, 'n b c h w -> (n b) c h w')
            grid = make_grid(grid, nrow=int(sqrt(args.batchsize)))
            grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
            grid = Image.fromarray(grid.astype(np.uint8))
            grid.save(os.path.join(sample_output_dir, 'grid.png'))
            
            # Final memory cleanup
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\nError processing {img_path}: {str(e)}")
            continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path', default="ldm/models/ldm/inpainting_big/config_LAKERED.yaml", 
                       help='Path to YAML config')
    parser.add_argument('--model_path', default="ckpt/LAKERED.ckpt", 
                       help='Path to model checkpoint')
    parser.add_argument('--dataset_dir', default="LAKERED_DATASET", 
                       help='Path to dataset directory')
    parser.add_argument('--output_dir', default="results", 
                       help='Directory to save results')
    parser.add_argument('--batchsize', type=int, default=9, 
                       help='Batch size for sampling')
    parser.add_argument('--isReplace', action='store_true', 
                       help='Whether to replace the foreground')
    parser.add_argument('--dilate_kernel', type=int, default=2, 
                       help='Dilation kernel size')
    parser.add_argument('--Steps', type=int, default=50, 
                       help='Number of DDIM steps')
    
    args = parser.parse_args()
    print('Processing full dataset with args:')
    print(args)
    
    process_dataset(args) 