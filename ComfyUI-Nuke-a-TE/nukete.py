import os
import logging
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.functional import normalize
import importlib
import folder_paths
import node_helpers
import comfy
import comfy.model_management as mm
from comfy.model_management import get_torch_device
import comfy.diffusers_load
import comfy.sd
import comfy.utils
import comfy.clip_model
import comfy.supported_models_base
from comfy import sd1_clip
import comfy.text_encoders.t5
import comfy.samplers
import comfy.sample
import comfy.controlnet
import comfy.clip_vision
from comfy.cli_args import args
import comfy.sdxl_clip
import numpy as np
import safetensors.torch
import latent_preview
import random


class CLIPTextEncodeNUKE:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "clip": ("CLIP", {}),
                "nuke_clip": (["False", "True"], {"default": "False"}),
                "randn_clip": (["False", "True"], {"default": "False"}),
                "seed": ("INT", {"default": 425533035839474, "min": 0, "max": 0xffffffffffffffff}),
                "custom_embeds": (["False", "True"], {"default": "False"}),
                "embeds_idx": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1}),
                "embeds_path": ("STRING", {"default": "embeds/customembedding.pt"}),
            }
        }
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "zer0int/NUKE-a-TE"
    DESCRIPTION = "Nukes or randomizes text encoder input; loads custom embeddings for CLIP guidance."

    def encode(self, clip, text, nuke_clip, randn_clip, custom_embeds, embeds_path, embeds_idx, seed):
        tokens = clip.tokenize(text)
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)

        # cond is a tensor: [1,77,768], extracted from output
        cond = output.pop("cond")
        
        l2_norm = cond.norm(p=2)
        normalized = cond / l2_norm 
        
        #print(f"Normalized Embedding: {normalized}")
        #print(f"L2 Norm After Normalization: {normalized.norm(p=2)}")

        if custom_embeds == "True":
            if not os.path.exists(embeds_path):
                raise FileNotFoundError("Custom embedding file not found")

            custom_embedding = torch.load(embeds_path, map_location="cpu")
            if custom_embedding.dim() != 2 or custom_embedding.shape[1] != 768:
                raise ValueError("Custom embedding must have shape [N,768]")

            num_embeddings = custom_embedding.size(0)
            if embeds_idx < 0 or embeds_idx >= num_embeddings:
                embeds_idx = 0

            # Extract [1,768] embedding
            chosen_embedding = custom_embedding[embeds_idx:embeds_idx+1].float()
            original_scale = chosen_embedding.norm(p=2, dim=-1, keepdim=True)                 
            modified_clip_embeddings = chosen_embedding / original_scale               
            
            # Convert to [1,77,768]
            selected_embedding = modified_clip_embeddings.unsqueeze(1).repeat(1,77,1)

            # If the original cond had a different batch dimension, match it:
            if cond.shape[0] != selected_embedding.shape[0]:
                factor = cond.shape[0] // selected_embedding.shape[0]
                if factor * selected_embedding.shape[0] == cond.shape[0]:
                    selected_embedding = selected_embedding.repeat(factor,1,1)

            # Replace the cond tensor directly
            cond = selected_embedding

        l2_norm = cond.norm(p=2)
        normalized = cond / l2_norm 
        
        #print(f"Post- Normalized Embedding: {normalized}")
        #print(f" Post- L2 Norm After Normalization: {normalized.norm(p=2)}")

        if randn_clip == "True":
            torch.manual_seed(seed)
            cond = torch.randn_like(cond)

        if nuke_clip == "True":
            cond = torch.zeros_like(cond)

        # cond is a tensor, output is a dict
        return ([[cond, output]], )


# This SDXL crazy tensorial slicing / injection 'heuristic scheme' was thought up by OpenAI's o1 after we couldn't 'find' CLIP-L.
# The result is 'unexpected'; a piece of o1's tensorial art. I am leaving it as-is, without any changes. 100% AI code art below. ~zer0int
class CLIPTextEncodeSDXLNUKE:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"tooltip": "Checkpoint to load."}),
                "clip_name1": (folder_paths.get_filename_list("clip"), {"tooltip": "First CLIP model to load for debugging."}),
                "clip_name2": (folder_paths.get_filename_list("clip"), {"tooltip": "Second CLIP model (CLIP-L) to load for debugging."}),
                "width": ("INT", {"default": 1024, "min": 0, "max": 8192}),
                "height": ("INT", {"default": 1024, "min": 0, "max": 8192}),
                "crop_w": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "crop_h": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "target_width": ("INT", {"default": 1024, "min": 0, "max": 8192}),
                "target_height": ("INT", {"default": 1024, "min": 0, "max": 8192}),
                "text_g": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "text_l": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "nuke_clip": (["False", "True"], {"default": "False"}),
                "randn_clip": (["False", "True"], {"default": "False"}),
                "nuke_bigclip": (["False", "True"], {"default": "False"}),
                "randn_bigclip": (["False", "True"], {"default": "False"}),
                "seed": ("INT", {"default": 425533035839474, "min": 0, "max": 0xffffffffffffffff}),
                "custom_embeds": (["False", "True"], {"default": "False"}),
                "embeds_idx": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1}),
                "embeds_path": ("STRING", {"default": "embeds/customembedding.pt", "multiline": False}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "CONDITIONING",)
    FUNCTION = "load_and_encode"
    CATEGORY = "zer0int/NUKE-a-TE"
    DESCRIPTION = "Loads a checkpoint (model, clip, vae) and two separate CLIPs for debugging, then encodes text using the loaded checkpoint's CLIP, applies nuking, randn, and custom embeddings."

    def load_checkpoint(self, ckpt_name):
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings")
        )
        model, clip_from_ckpt, vae = out[:3]
        return model, clip_from_ckpt, vae

    def load_clip_model(self, clip_name):
        clip_path = folder_paths.get_full_path_or_raise("clip", clip_name)
        clip_type = 1  # STABLE_DIFFUSION
        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type
        )
        return clip

    def load_and_encode(self, ckpt_name, clip_name1, clip_name2, width, height, crop_w, crop_h, target_width, target_height,
                        text_g, text_l, nuke_clip, randn_clip, nuke_bigclip, randn_bigclip,
                        custom_embeds, embeds_path, embeds_idx, seed):

        model, clip, vae = self.load_checkpoint(ckpt_name)
        #print("=== DEBUG: LOADED CHECKPOINT ===")
        #print("Model type:", type(model))
        #print("Clip from checkpoint type:", type(clip))
        #print("VAE type:", type(vae))

        # Debug load separate CLIPs
        #print("Loading clip_name1:", clip_name1)
        debug_clip_g = self.load_clip_model(clip_name1)
        #print("Loading clip_name2:", clip_name2)
        debug_clip_l = self.load_clip_model(clip_name2)

        test_tokens_g = debug_clip_g.tokenize("Hello from G")
        test_tokens_l = debug_clip_l.tokenize("Hello from L")
        #print("Debug CLIP-G tokens:", test_tokens_g)
        #print("Debug CLIP-L tokens:", test_tokens_l)

        tokens_g = clip.tokenize(text_g)  
        tokens_l = clip.tokenize(text_l)

        #print("tokens_g['g'] type:", type(tokens_g['g']))
        #print("tokens_l['l'] type:", type(tokens_l['l']))
        #if len(tokens_g['g']) > 0:
        #    print("First token in g:", tokens_g['g'])
        #if len(tokens_l['l']) > 0:
        #    print("First token in l:", tokens_l['l'])

        empty_tokens = clip.tokenize("")
        empty_g = empty_tokens["g"]
        empty_l = empty_tokens["l"]

        g_len = len(tokens_g["g"])
        l_len = len(tokens_l["l"])
        if g_len != l_len:
            print(f"Length mismatch: g_len={g_len}, l_len={l_len}, padding...")
            while len(tokens_l["l"]) < len(tokens_g["g"]):
                tokens_l["l"].extend(empty_l)
            while len(tokens_l["l"]) > len(tokens_g["g"]):
                tokens_g["g"].extend(empty_g)

        tokens = {"g": tokens_g['g'], "l": tokens_l['l']}
        #print("After ensuring equal length:")
        #print("len(tokens['g']):", len(tokens['g']))
        #print("len(tokens['l']):", len(tokens['l']))

        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        #print("cond shape:", cond.shape, "dtype:", cond.dtype)
        #print("pooled shape:", pooled.shape, "dtype:", pooled.dtype)
        #if cond.numel() > 0:
        #    print("cond sample (first token embedding):", cond[0,0,:8])
        #if pooled.numel() > 0:
        #    print("pooled sample (first few elems):", pooled[0,:8])

        total_dim = pooled.shape[-1]
        #print("Detected pooled dimension:", total_dim)

        def try_custom_embedding(p_l, scenario=""):
            if custom_embeds == "True":
                print(f"Trying to load custom embedding from {embeds_path} for scenario {scenario}")
                if not os.path.exists(embeds_path):
                    raise FileNotFoundError(f"Custom embedding file not found: {embeds_path}")

                custom_embedding = torch.load(embeds_path, map_location="cpu").float()
                print("Custom embedding loaded. Shape:", custom_embedding.shape, "dtype:", custom_embedding.dtype)
                if custom_embedding.dim() == 2 and custom_embedding.shape[1] == 768:
                    num_embeddings = custom_embedding.size(0)
                    use_idx = embeds_idx
                    if use_idx < 0 or use_idx >= num_embeddings:
                        use_idx = 0
                    selected_embedding = custom_embedding[use_idx:use_idx+1]
                    print("Selected embedding index:", use_idx, "shape:", selected_embedding.shape)
                    if selected_embedding.shape == (1, 768):
                        print("Successfully matched L dimension, replacing pooled_l with selected embedding.")
                        return selected_embedding
                    else:
                        print("selected_embedding shape not (1,768). Skipping.")
                else:
                    print("Custom embedding not 768-dim or unexpected shape:", custom_embedding.shape)
            return p_l

        def apply_bigclip_ops(cg, pg):
            if cg is not None and pg is not None:
                if randn_bigclip == "True":
                    torch.manual_seed(seed)
                    cg = torch.randn_like(cg)
                    pg = torch.randn_like(pg)
                if nuke_bigclip == "True":
                    cg = torch.zeros_like(cg)
                    pg = torch.zeros_like(pg)
            return cg, pg

        def apply_clip_ops(cl_, pl_):
            if randn_clip == "True":
                torch.manual_seed(seed)
                cl_ = torch.randn_like(cl_)
                pl_ = torch.randn_like(pl_)
            if nuke_clip == "True":
                cl_ = torch.zeros_like(cl_)
                pl_ = torch.zeros_like(pl_)
            return cl_, pl_

        if total_dim == 2048:
            # SDXL scenario
            g_dim = 1280
            l_dim = 768
            cond_g = cond[..., :g_dim]
            cond_l = cond[..., g_dim:g_dim+l_dim]
            pooled_g = pooled[..., :g_dim]
            pooled_l = pooled[..., g_dim:g_dim+l_dim]

            pooled_l = try_custom_embedding(pooled_l, scenario="SDXL")

            cond_g, pooled_g = apply_bigclip_ops(cond_g, pooled_g)
            cond_l, pooled_l = apply_clip_ops(cond_l, pooled_l)

            cond = torch.cat([cond_g, cond_l], dim=-1)
            pooled = torch.cat([pooled_g, pooled_l], dim=-1)

        elif total_dim == 768:
            # Pure L scenario
            pooled = try_custom_embedding(pooled, scenario="Pure L")

            cond, pooled = apply_clip_ops(cond, pooled)
            # If needed, apply bigclip ops to entire embedding as well
            if randn_bigclip == "True" or nuke_bigclip == "True":
                cond, pooled = apply_clip_ops(cond, pooled)

        else:
            # Unknown dimension scenario
            print("Unknown dimension scenario. Attempting heuristic for custom embedding if 768-dim embedding is requested.")

            if custom_embeds == "True":
                # If total_dim >= 768, try last 768 dims as L
                if total_dim >= 768:
                    print(f"Attempting heuristic: carving out last 768 dims as L portion from total_dim={total_dim}.")
                    l_dim = 768
                    g_dim = total_dim - l_dim
                    cond_g = cond[..., :g_dim] if g_dim > 0 else None
                    cond_l = cond[..., g_dim:] if g_dim > 0 else cond
                    pooled_g = pooled[..., :g_dim] if g_dim > 0 else None
                    pooled_l = pooled[..., g_dim:] if g_dim > 0 else pooled

                    print("Heuristic L portion shape:", pooled_l.shape)
                    pooled_l = try_custom_embedding(pooled_l, scenario=f"Unknown dim {total_dim} with L=768")

                    # Apply ops
                    cond_g, pooled_g = apply_bigclip_ops(cond_g, pooled_g) if g_dim > 0 else (cond_g, pooled_g)
                    cond_l, pooled_l = apply_clip_ops(cond_l, pooled_l)

                    # Recombine
                    if g_dim > 0:
                        cond = torch.cat([cond_g, cond_l], dim=-1)
                        pooled = torch.cat([pooled_g, pooled_l], dim=-1)
                    else:
                        # total_dim < 768 would skip here, but we already checked total_dim >= 768
                        cond = cond_l
                        pooled = pooled_l
                else:
                    print("total_dim < 768, cannot carve out L portion. Skipping custom embedding.")
                    # Just apply ops to entire tensor
                    if randn_bigclip == "True" or randn_clip == "True":
                        torch.manual_seed(seed)
                        cond = torch.randn_like(cond)
                        pooled = torch.randn_like(pooled)
                    if nuke_bigclip == "True" or nuke_clip == "True":
                        cond = torch.zeros_like(cond)
                        pooled = torch.zeros_like(pooled)
            else:
                # No custom embedding requested
                if randn_bigclip == "True" or randn_clip == "True":
                    torch.manual_seed(seed)
                    cond = torch.randn_like(cond)
                    pooled = torch.randn_like(pooled)
                if nuke_bigclip == "True" or nuke_clip == "True":
                    cond = torch.zeros_like(cond)
                    pooled = torch.zeros_like(pooled)

        #print("Final cond shape:", cond.shape)
        #print("Final pooled shape:", pooled.shape)

        conditioning = [[cond, {
            "pooled_output": pooled,
            "width": width,
            "height": height,
            "crop_w": crop_w,
            "crop_h": crop_h,
            "target_width": target_width,
            "target_height": target_height
        }]]

        return (model, clip, vae, conditioning)


class CLIPTextEncodeFluxNUKE:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", ),
                "clip_l": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "t5xxl": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "guidance": ("FLOAT", {"default": 11.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "randn_t5": (["False", "True"], {"default": "False"}),
                "nuke_t5": (["False", "True"], {"default": "True"}),
                "nuke_clip": (["False", "True"], {"default": "False"}), 
                "randn_clip": (["False", "True"], {"default": "False"}),
                "seed": ("INT", {"default": 425533035839474, "min": 0, "max": 0xffffffffffffffff}),
                "custom_embeds": (["False", "True"], {"default": "False"}),            
                "embeds_idx": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1}),
                "embeds_path": ("STRING", {"default": "path/to/embedding.pt", "multiline": False}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    OUTPUT_TOOLTIPS = ("Nukes or randomizes text encoder input; loads custom embeddings for CLIP guidance.",)
    FUNCTION = "encode"

    CATEGORY = "zer0int/NUKE-a-TE"
    DESCRIPTION = "Nukes or randomizes text encoder input; loads custom embeddings for CLIP guidance." 

    def encode(self, clip, clip_l, t5xxl, guidance, randn_t5, nuke_t5, nuke_clip, randn_clip, custom_embeds, embeds_path, embeds_idx, seed):   

        if custom_embeds == "True":
            tokens = clip.tokenize(clip_l)
            tokens["t5xxl"] = clip.tokenize(t5xxl)["t5xxl"]

            output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
            clip_embeddings = output["pooled_output"]                     
            modified_clip_embeddings = clip_embeddings

            if not os.path.exists(embeds_path):
                raise FileNotFoundError(f"Custom embedding file not found: {embeds_path}")

            custom_embedding = torch.load(embeds_path, map_location="cpu").float()
            if custom_embedding.dim() != 2 or custom_embedding.shape[1] != 768:
                raise ValueError(
                    f"Custom embedding must have shape [batch_size, 768]. Found: {custom_embedding.shape}"
                )

            num_embeddings = custom_embedding.size(0)
            if embeds_idx >= num_embeddings or embeds_idx < 0:
                embeds_idx = 0
                
            selected_embedding = custom_embedding[embeds_idx:embeds_idx + 1].float()
            modified_clip_embeddings = selected_embedding
            output["pooled_output"] = modified_clip_embeddings
            
            if randn_t5 == "True":
                torch.manual_seed(seed)
                output["cond"] = torch.randn_like(output["cond"])

            if nuke_t5 == "True":
                output["cond"] = torch.zeros_like(output["cond"])                  

            if randn_clip == "True":
                torch.manual_seed(seed)
                output["pooled_output"] = torch.randn_like(output["pooled_output"]) 

            if nuke_clip == "True":
                output["pooled_output"] = torch.zeros_like(output["pooled_output"]) 
            
            output["guidance"] = guidance

        else:
            tokens = clip.tokenize(clip_l)
            tokens["t5xxl"] = clip.tokenize(t5xxl)["t5xxl"]

            output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)

            if randn_t5 == "True":
                torch.manual_seed(seed)
                output["cond"] = torch.randn_like(output["cond"])

            if nuke_t5 == "True":
                output["cond"] = torch.zeros_like(output["cond"])                  

            if randn_clip == "True":
                torch.manual_seed(seed)
                output["pooled_output"] = torch.randn_like(output["pooled_output"])

            if nuke_clip == "True":
                output["pooled_output"] = torch.zeros_like(output["pooled_output"]) 
            
            output["guidance"] = guidance

        cond = output.pop("cond", None)
        return ([[cond, output]], )

        
        
# Need this so we don't need to load all of o1's madness for the negative prompt
class CLIPTextEncodeSDXLminiNUKE:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "width": ("INT", {"default": 1024.0, "min": 0, "max": 8192}),
            "height": ("INT", {"default": 1024.0, "min": 0, "max": 8192}),
            "crop_w": ("INT", {"default": 0, "min": 0, "max": 8192}),
            "crop_h": ("INT", {"default": 0, "min": 0, "max": 8192}),
            "target_width": ("INT", {"default": 1024.0, "min": 0, "max": 8192}),
            "target_height": ("INT", {"default": 1024.0, "min": 0, "max": 8192}),
            "text_g": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "text_l": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "clip": ("CLIP", ),
            "nuke_clip": (["False", "True"], {"default": "False"}),
            "randn_clip": (["False", "True"], {"default": "False"}),
            "nuke_bigclip": (["False", "True"], {"default": "False"}),
            "randn_bigclip": (["False", "True"], {"default": "False"}),
            "seed": ("INT", {"default": 425533035839474, "min": 0, "max": 0xffffffffffffffff}),
        }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "zer0int/NUKE-a-TE"
    DESCRIPTION = "Nukes or randomizes text encoder input; loads custom embeddings for CLIP guidance."

    def encode(self, clip, width, height, crop_w, crop_h, target_width, target_height, text_g, text_l, nuke_clip, randn_clip, randn_bigclip, nuke_bigclip, seed):

        # Tokenize
        tokens_g = clip.tokenize(text_g)  
        tokens_l = clip.tokenize(text_l)

        # Ensure equal length
        empty_tokens = clip.tokenize("")
        empty_g = empty_tokens["g"]
        empty_l = empty_tokens["l"]

        g_len = len(tokens_g["g"])
        l_len = len(tokens_l["l"])
        
        if g_len != l_len:
            print(f"Length mismatch: g_len={g_len}, l_len={l_len}, padding...")
            while len(tokens_l["l"]) < len(tokens_g["g"]):
                tokens_l["l"].extend(empty_l)
            while len(tokens_l["l"]) > len(tokens_g["g"]):
                tokens_g["g"].extend(empty_g)

        tokens = {"g": tokens_g['g'], "l": tokens_l['l']}

        # Encode
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        total_dim = pooled.shape[-1]
        
        if total_dim == 2048:
            # SDXL scenario
            g_dim = 1280
            l_dim = 768
            cond_g = cond[..., :g_dim]
            cond_l = cond[..., g_dim:g_dim+l_dim]
            pooled_g = pooled[..., :g_dim]
            pooled_l = pooled[..., g_dim:g_dim+l_dim]

            # Randomization/nuking
            if randn_bigclip == "True":
                torch.manual_seed(seed)
                cond_g = torch.randn_like(cond_g)
                pooled_g = torch.randn_like(pooled_g)
            if nuke_bigclip == "True":
                cond_g = torch.zeros_like(cond_g)
                pooled_g = torch.zeros_like(pooled_g)

            if randn_clip == "True":
                torch.manual_seed(seed)
                cond_l = torch.randn_like(cond_l)
                pooled_l = torch.randn_like(pooled_l)
            if nuke_clip == "True":
                cond_l = torch.zeros_like(cond_l)
                pooled_l = torch.zeros_like(pooled_l)

            # Recombine
            cond = torch.cat([cond_g, cond_l], dim=-1)
            pooled = torch.cat([pooled_g, pooled_l], dim=-1)

        else:
            # Unknown dimension scenario (e.g., 1280)
            # If user wants to nuke or randn, apply to entire tensor:
            if randn_bigclip == "True" or randn_clip == "True":
                torch.manual_seed(seed)
                cond = torch.randn_like(cond)
                pooled = torch.randn_like(pooled)
            if nuke_bigclip == "True" or nuke_clip == "True":
                cond = torch.zeros_like(cond)
                pooled = torch.zeros_like(pooled)

        return ([[cond, {
            "pooled_output": pooled,
            "width": width,
            "height": height,
            "crop_w": crop_w,
            "crop_h": crop_h,
            "target_width": target_width,
            "target_height": target_height
        }]], )
