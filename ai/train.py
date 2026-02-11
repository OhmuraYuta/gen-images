
import os
import argparse
import itertools
import math
import random
from pathlib import Path

import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from huggingface_hub import HfFolder, Repository, whoami
from peft import LoraConfig, get_peft_model
from PIL import Image
from PIL.ImageOps import exif_transpose
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0")

def parse_args():
    parser = argparse.ArgumentParser(description="Simple SDXL LoRA training script.")
    parser.add_argument("--instance_data_dir", type=str, default=None, required=True)
    parser.add_argument("--output_dir", type=str, default="lora-sdxl-trained", required=True)
    parser.add_argument("--instance_prompt", type=str, default="a photo of sks dog", required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=1e-4) # Higher for LoRA
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4) # Increase effective batch size
    parser.add_argument("--optimizer", type=str, default="adamw_8bit", choices=["adamw", "adamw_8bit"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    
    args = parser.parse_args()
    return args

class DreamBoothDataset(torch.utils.data.Dataset):
    def __init__(self, instance_data_root, instance_prompt, tokenizer_one, tokenizer_two, size=1024):
        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance data root doesn't exist.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.instance_prompt = instance_prompt
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two
        self.size = size

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.instance_images_path)

    def __getitem__(self, index):
        example = {}
        image = Image.open(self.instance_images_path[index % len(self.instance_images_path)])
        image = exif_transpose(image)
        if not image.mode == "RGB":
            image = image.convert("RGB")
            
        example["pixel_values"] = self.image_transforms(image)

        # SDXL uses two text encoders
        # Simple implementation: just tokenize the instance prompt
        # In a full detailed training script we might need to handle pooled outputs etc.
        # For simplicity in this fix, we tokenize twice
        example["input_ids_one"] = self.tokenizer_one(
            self.instance_prompt, padding="max_length", truncation=True, max_length=self.tokenizer_one.model_max_length, return_tensors="pt"
        ).input_ids[0]
        
        example["input_ids_two"] = self.tokenizer_two(
            self.instance_prompt, padding="max_length", truncation=True, max_length=self.tokenizer_two.model_max_length, return_tensors="pt"
        ).input_ids[0]

        return example

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    
    input_ids_one = torch.stack([example["input_ids_one"] for example in examples])
    input_ids_two = torch.stack([example["input_ids_two"] for example in examples])
    
    return {
        "pixel_values": pixel_values,
        "input_ids_one": input_ids_one,
        "input_ids_two": input_ids_two
    }

def main():
    args = parse_args()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16" 
    )

    if args.seed is not None:
        set_seed(args.seed)

    # Load models
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    
    # Load Tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer", use_fast=False)
    tokenizer_two = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer_2", use_fast=False)

    # Load Scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # Load UNet
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float16) # Load in fp16 base
    
    # Load VAE (Frozen)
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

    # Load Text Encoders
    from transformers import CLIPTextModel, CLIPTextModelWithProjection
    text_encoder_one = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.float16)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(model_id, subfolder="text_encoder_2", torch_dtype=torch.float16)

    # Freeze Base Models
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)

    # Move to device
    weight_dtype = torch.float16
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    # --- LoRA Config ---
    # We only train UNet LoRA for simplicity
    unet_lora_config = LoraConfig(
        r=4, # Rank, low for quick training
        lora_alpha=4,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)

    # FIX: Cast trainable parameters (LoRA) to float32 for mixed precision stability
    # This prevents "ValueError: Attempting to unscale FP16 gradients"
    for param in unet.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    # Data Loader
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        tokenizer_one=tokenizer_one,
        tokenizer_two=tokenizer_two,
        size=args.resolution,
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=1,
    )
    
    # Prepare with Accelerator
    # Note: We only prepare optimizer and dataloader. UNet is manually handled or could be prepared if fully trainable.
    # For LoRA attached to UNet, we usually need to prepare the model too if using accelerate's save/load seamlessly,
    # but since unet is large, we might face issues.
    # Let's clean up: prepare everything that needs grad or device placement if not handled.
    # But text encoders/vae are frozen.
    
    # Correct way for LoRA usually involves `peft` handling, but here 'unet' has adapter.
    # We pass 'unet' to prepare to potential wrap it.
    unet, train_dataloader = accelerator.prepare(unet, train_dataloader)

    # Make sure only LoRA layers are trainable
    # We fetch params AFTER prepare, just to be safe if model was wrapped
    lora_layers = list(filter(lambda p: p.requires_grad, unet.parameters()))
    
    # Optimization
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.optimizer == "adamw_8bit":
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
        except ImportError:
            raise ImportError("Please install bitsandbytes to use adamw_8bit optimizer")
    else:
        optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        lora_layers,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )
    
    optimizer = accelerator.prepare(optimizer)

    # Training Loop
    num_update_steps_per_epoch = len(train_dataloader)
    
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        
    global_step = 0
    
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    print(f"Designed Training Steps: {args.max_train_steps}")
    
    unet.train()
    
    # Helper to compute text embeddings (since we don't train TE)
    def compute_text_embeddings(input_ids_one, input_ids_two):
        with torch.no_grad():
            prompt_embeds_list = []
            for input_ids, text_encoder in zip([input_ids_one, input_ids_two], [text_encoder_one, text_encoder_two]):
                prompt_embeds = text_encoder(input_ids, output_hidden_states=True)
                # We need pooled output from 2nd encoder
                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds_list.append(prompt_embeds.hidden_states[-2]) # Penultimate layer
                
            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
            pooled_prompt_embeds = prompt_embeds_list[1] # Logic is complex for SDXL full implementation
            # Simplified for now: getting the concat embeddings. 
            # SDXL needs 'add_text_embeds' and 'add_time_ids'.
            
            # RE-CHECK: SDXL Text Encoding is complex involving 2 encoders and pooling.
            # Using a simplified pipeline approach or just using what we can.
            # Given complexity, we will skip advanced Aspect Ratio bucketing/Time IDs conditioning 
            # and stick to a simpler "DreamBooth" style conditioning if possible, 
            # OR use the standard SDXL conditioning approach.
            pass
            # Since implementing full SDXL conditioning from scratch in a simple script is risky for 1-shot,
            # We will rely on the fact that we can just feed prompt_embeds if valid.
            
            # Let's USE PRE-COMPUTED EMBEDDINGS strategy implicitly by encoding inside loop?
            return None # Placeholder

    # Re-impl strategy: Stick to simple "Unwrap" or just simple encoding in loop.
    # Actually, simpler to just run forward pass? No, VAE/Text Encoder outputs needed.
    
    # Let's simplify conditioning: just use standard pipeline components logic inline? 
    # Or even better: Use `compute_time_ids` similar to difusers scripts.
    
    # Pre-calculating empty text embeds etc is safer. 
    # But here, we will iterate.
    
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # 1. Convert images to latents
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # 2. Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                
                # 3. Add noise
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # 4. Get Text Embeddings
                # SDXL specific:
                with torch.no_grad():
                     prompt_embeds_1 = text_encoder_one(batch["input_ids_one"], output_hidden_states=True)
                     prompt_embeds_2 = text_encoder_two(batch["input_ids_two"], output_hidden_states=True)
                     
                     # Check shape/tuple
                     # text_encoder output is usually BaseModelOutputWithPoolingAndCrossAttentions
                     
                     hidden_states_1 = prompt_embeds_1.hidden_states[-2]
                     hidden_states_2 = prompt_embeds_2.hidden_states[-2]
                     
                     prompt_embeds = torch.cat([hidden_states_1, hidden_states_2], dim=-1)
                     
                     # Pooled Output (from 2nd)
                     pooled_prompt_embeds = prompt_embeds_2.text_embeds # or pooler_output
                     # NOTE: SDXL CLIPTextModelWithProjection has text_embeds
                
                # 5. Add Time IDs (Micro-conditioning)
                # Size of original image (1024, 1024), crop (0,0), target (1024,1024)
                def compute_time_ids(original_size, crops_coords_top_left=(0,0), target_size=None):
                    if target_size is None: target_size = original_size
                    add_time_ids = list(original_size + crops_coords_top_left + target_size)
                    add_time_ids = torch.tensor([add_time_ids], dtype=weight_dtype, device=accelerator.device)
                    return add_time_ids
                
                add_time_ids = compute_time_ids((args.resolution, args.resolution))
                add_time_ids = add_time_ids.repeat(bsz, 1)

                # 6. Predict noise
                # UNet Input: noisy_latents, timesteps, encoder_hidden_states=prompt_embeds, added_cond_kwargs
                added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}
                
                target = noise
                
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False
                )[0] # Extract sample

                # 7. Loss
                if noise_scheduler.config.prediction_type == "epsilon":
                    loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else: 
                     # v_prediction
                    loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_layers, 1.0)
                    
                optimizer.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                if global_step % 10 == 0:
                     print(f"Steps: {global_step}/{args.max_train_steps} loss: {loss.item() if loss is not None else 0:.4f}", flush=True)
                
            if global_step >= args.max_train_steps:
                break
                
    # Save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("Training finished! Saving LoRA weights...")
        
        # Unwrap and save adapter
        # Unwrap model
        unet = accelerator.unwrap_model(unet)
        
        # Save ONLY LoRA weights (filtering by "lora" in key)
        # We need to prepend "unet." prefix so that pipe.load_lora_weights identifies them correctly
        unet_lora_state_dict = {f"unet.{k}": v for k, v in unet.state_dict().items() if "lora" in k}
        
        from safetensors.torch import save_file
        save_path = os.path.join(args.output_dir, "pytorch_lora_weights.safetensors")
        save_file(unet_lora_state_dict, save_path)
        print(f"Saved LoRA weights to {save_path}")
        
        # We need "pytorch_lora_weights.safetensors" for our loading logic in main.py?
        # main.py looks for ANY .safetensors. PEFT standard is `adapter_model.safetensors`
        # Let's ensure we return a safetensors file.
        
        # Post-processing: Rename or ensure main.py finds it.
        # main.py: "if f.endswith('.safetensors')"
        # save_pretrained usually saves `adapter_model.safetensors` if safe_serialization=True is default or passed.
        # It seems PEFT defualts to .bin (PyTorch) sometimes.
        # Force safetensors or just rename.
        pass

if __name__ == "__main__":
    main()
