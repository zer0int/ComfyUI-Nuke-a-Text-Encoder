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
import torch.nn.functional as F

class CLIPTextEncodeDiffVec:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "clip": ("CLIP", {}),
                "clip_l_minusvec": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "Negative concepts to subtract from CLIP embedding, separated by | (e.g. cat|dog)."}),
                "clip_l_addvec": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "Positive concepts to add from CLIP embedding, separated by | (e.g. cat|dog)."}),
                "clip_minusvec_alpha": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step": 0.001, "tooltip": "Scaling for difference vector (subtract mode only)"}),
                "clip_addvec_alpha": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step": 0.001, "tooltip": "Scaling for addition vector (add mode only)"}),
                "simple_neg": (["False", "True"], {"default": "False"}),
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
    CATEGORY = "zer0int/DiffVec"
    DESCRIPTION = "Token-wise difference/addition vectors for CLIP sequence embedding. Always returns proper [1, 77, d_model] sequence as conditioning."

    @staticmethod
    def sequence_vector_subtract(base_emb, negative_embs, alpha=1.0):
        """
        Subtracts the mean negative vector(s) from base sequence, token-wise.
        All tensors [1, 77, d_model], negatives can be [N, 77, d_model]
        """
        if negative_embs is None or negative_embs.shape[0] == 0:
            return base_emb
        neg_mean = negative_embs.mean(dim=0, keepdim=True)
        return base_emb - alpha * neg_mean

    @staticmethod
    def sequence_vector_subtract_simple(base_emb, negative_embs, alpha=1.0):
        """
        Simple sum subtraction per token position.
        """
        return base_emb - alpha * negative_embs.sum(dim=0, keepdim=True)

    @staticmethod
    def sequence_vector_add(base_emb, positive_embs, alpha=1.0):
        """
        Tokenwise addition of mean positive concept vector(s).
        """
        return base_emb + alpha * positive_embs.mean(dim=0, keepdim=True)

    @staticmethod
    def rescale_to_match_base(final_emb, base_emb):
        """
        Scales the final embedding so its total L2 norm matches base_emb.
        Both tensors should be [1, 77, d_model].
        """
        final_norm = final_emb.norm(dim=[-1, -2], keepdim=True)
        base_norm = base_emb.norm(dim=[-1, -2], keepdim=True)
        scale = (base_norm / (final_norm + 1e-6))
        return final_emb * scale

    def encode(self, clip, clip_l_minusvec, clip_l_addvec, clip_minusvec_alpha, clip_addvec_alpha, simple_neg, text, nuke_clip, randn_clip, custom_embeds, embeds_path, embeds_idx, seed):
        tokens = clip.tokenize(text)
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        base_emb = output["cond"].clone()  # [1, 77, d_model]
        original_clip_embedding = base_emb.clone()
        cond = base_emb

        if custom_embeds == "True":
            if not os.path.exists(embeds_path):
                raise FileNotFoundError(f"Custom embedding file not found: {embeds_path}")
            custom_embedding = torch.load(embeds_path, map_location="cpu").float()  # [num_embeds, d_model]
            if custom_embedding.dim() != 2 or custom_embedding.shape[1] not in (768, 1024):
                raise ValueError(f"Custom embedding must have shape [num_embeds, d_model]. Found: {custom_embedding.shape}")
            num_embeddings = custom_embedding.size(0)
            if embeds_idx >= num_embeddings or embeds_idx < 0:
                embeds_idx = 0
            chosen_embedding = custom_embedding[embeds_idx:embeds_idx+1].float()  # [1, d_model]
            # Broadcast to [1, 77, d_model]
            cond = chosen_embedding.unsqueeze(1).repeat(1, 77, 1)
            original_clip_embedding = cond.clone()

        # --- CLIP minusvec (subtract negative concepts, token-wise) ---
        if clip_l_minusvec.strip():
            negative_prompts = [s.strip() for s in clip_l_minusvec.split('|') if s.strip()]
            if negative_prompts:
                negative_embs = []
                for neg in negative_prompts:
                    neg_seq = clip.encode_from_tokens(clip.tokenize(neg), return_pooled=True, return_dict=True)["cond"]
                    negative_embs.append(neg_seq)
                negative_embs = torch.cat(negative_embs, dim=0)  # [N, 77, d_model]
                if simple_neg == "True":
                    cond = self.sequence_vector_subtract_simple(cond, negative_embs, clip_minusvec_alpha)
                else:
                    cond = self.sequence_vector_subtract(cond, negative_embs, clip_minusvec_alpha)
                cond = self.rescale_to_match_base(cond, original_clip_embedding)

        # --- CLIP addvec (add positive concepts, token-wise) ---
        if clip_l_addvec.strip():
            positive_prompts = [s.strip() for s in clip_l_addvec.split('|') if s.strip()]
            if positive_prompts:
                positive_embs = []
                for pos in positive_prompts:
                    pos_seq = clip.encode_from_tokens(clip.tokenize(pos), return_pooled=True, return_dict=True)["cond"]
                    positive_embs.append(pos_seq)
                positive_embs = torch.cat(positive_embs, dim=0)  # [N, 77, d_model]
                cond = self.sequence_vector_add(cond, positive_embs, clip_addvec_alpha)
                cond = self.rescale_to_match_base(cond, original_clip_embedding)

        # --- Nuke or randomize the embedding ---
        if randn_clip == "True":
            torch.manual_seed(seed)
            cond = torch.randn_like(cond)
        if nuke_clip == "True":
            cond = torch.zeros_like(cond)

        output["cond"] = cond  # [1, 77, d_model]
        return ([[cond, output]], )


class CLIPTextEncodeSDXLDiffVec:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"tooltip": "Checkpoint to load."}),
                "clip_l_minusvec": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "Negative concepts to subtract from CLIP-L, separated by |."}),
                "clip_l_addvec": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "Positive concepts to add to CLIP-L, separated by |."}),
                "clip_g_minusvec": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "Negative concepts to subtract from CLIP-G, separated by |."}),
                "clip_g_addvec": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "Positive concepts to add to CLIP-G, separated by |."}),
                "clip_l_minusvec_alpha": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step": 0.001}),
                "clip_l_addvec_alpha": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step": 0.001}),
                "clip_g_minusvec_alpha": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step": 0.001}),
                "clip_g_addvec_alpha": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step": 0.001}),
                "simple_neg": (["False", "True"], {"default": "False"}),
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
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "CONDITIONING",)
    FUNCTION = "load_and_encode"
    CATEGORY = "zer0int/DiffVec"
    DESCRIPTION = "Loads a checkpoint (model, clip, vae), encodes text using the loaded checkpoint's CLIP, and applies custom CLIP-L/G vector arithmetic."

    @staticmethod
    def vector_subtract(base_emb, negative_embs, alpha=1.0):
        if negative_embs is None or negative_embs.shape[0] == 0:
            return base_emb
        Q, _ = torch.linalg.qr(negative_embs.T, mode='reduced')
        proj = Q @ (Q.T @ base_emb.squeeze(0))
        emb_out = base_emb.squeeze(0) - alpha * proj
        return emb_out.unsqueeze(0)

    @staticmethod
    def vector_subtract_simple(base_emb, negative_embs, alpha=1.0):
        return base_emb - alpha * negative_embs.sum(dim=0, keepdim=True)

    @staticmethod
    def addition_vector_add(base_emb, positive_embs, alpha=1.0):
        return base_emb + alpha * positive_embs.sum(dim=0, keepdim=True)

    @staticmethod
    def rescale_to_match_base(final_emb, base_emb):
        final_norm = final_emb.norm(dim=-1, keepdim=True)
        base_norm = base_emb.norm(dim=-1, keepdim=True)
        scale = (base_norm / (final_norm + 1e-6))
        return final_emb * scale

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

    def _stack_embeddings(self, emb_list):
        # Robustly stack embeddings to [N, D]
        if not emb_list:
            return None
        emb_list = [
            e.unsqueeze(0) if e.ndim == 1 else e.view(-1, e.shape[-1]) if e.ndim > 2 else e
            for e in emb_list
        ]
        out = torch.cat(emb_list, dim=0)
        if out.ndim == 1:
            out = out.unsqueeze(0)
        return out

    def load_and_encode(self, ckpt_name, 
                        clip_l_minusvec, clip_l_addvec, clip_l_minusvec_alpha, clip_l_addvec_alpha, 
                        clip_g_minusvec, clip_g_addvec, clip_g_minusvec_alpha, clip_g_addvec_alpha,
                        simple_neg, width, height, crop_w, crop_h, target_width, target_height,
                        text_g, text_l, nuke_clip, randn_clip, nuke_bigclip, randn_bigclip,
                        seed):

        raise RuntimeError("This node doesn't work. Happy to take a pull request if you want to figure out how this checkpoint packing mess works. Otherwise: Sorry!")
        model, clip, vae = self.load_checkpoint(ckpt_name)

        # Tokenize and pad to match sequence
        tokens_g = clip.tokenize(text_g)
        tokens_l = clip.tokenize(text_l)
        empty_tokens = clip.tokenize("")
        empty_g = empty_tokens["g"]
        empty_l = empty_tokens["l"]

        g_len = len(tokens_g["g"])
        l_len = len(tokens_l["l"])
        if g_len != l_len:
            print(f"[DIFFVEC] Length mismatch: g_len={g_len}, l_len={l_len}, padding...")
            while len(tokens_l["l"]) < len(tokens_g["g"]):
                tokens_l["l"].extend(empty_l)
            while len(tokens_l["l"]) > len(tokens_g["g"]):
                tokens_g["g"].extend(empty_g)

        tokens = {"g": tokens_g['g'], "l": tokens_l['l']}

        # Get per-token embeddings [B, 77, 2048] (SDXL), or fallback for SD1
        encode_outputs = clip.encode_from_tokens(tokens, return_pooled=True)
        token_emb = encode_outputs[0] if isinstance(encode_outputs, (tuple, list)) else encode_outputs
        if token_emb.ndim == 2 and token_emb.shape[-1] == 2048:
            token_emb = token_emb.unsqueeze(0)  # [1, 77, 2048]

        pooled_g_tokens = token_emb[..., :1280]   # [B, 77, 1280]
        pooled_l_tokens = token_emb[..., 1280:]   # [B, 77, 768]
        pooled_g = pooled_g_tokens[:, 0, :]       # [B, 1280]
        pooled_l = pooled_l_tokens[:, 0, :]       # [B, 768]

        cond_g = pooled_g.clone()
        cond_l = pooled_l.clone()

        # ----------- CLIP-L Minusvec/Addvec -----------
        if clip_l_minusvec.strip():
            negative_prompts = [s.strip() for s in clip_l_minusvec.split('|') if s.strip()]
            negative_embs = []
            for neg in negative_prompts:
                neg_tokens = clip.tokenize(neg)
                neg_encode = clip.encode_from_tokens({"g": neg_tokens["g"], "l": neg_tokens["l"]}, return_pooled=True)
                neg_emb = neg_encode[0][..., 0, 1280:] if isinstance(neg_encode, (tuple, list)) else neg_encode[..., 0, 1280:]
                # Ensure always [1, 768]
                if neg_emb.ndim == 1:
                    neg_emb = neg_emb.unsqueeze(0)
                negative_embs.append(neg_emb)
            negative_embs = self._stack_embeddings(negative_embs)
            if negative_embs is not None and negative_embs.shape[0] > 0:
                if simple_neg == "True":
                    pooled_l = self.vector_subtract_simple(pooled_l, negative_embs, clip_l_minusvec_alpha)
                else:
                    pooled_l = self.vector_subtract(pooled_l, negative_embs, clip_l_minusvec_alpha)
                pooled_l = self.rescale_to_match_base(pooled_l, cond_l)
                cond_l = pooled_l.clone()

        if clip_l_addvec.strip():
            positive_prompts = [s.strip() for s in clip_l_addvec.split('|') if s.strip()]
            positive_embs = []
            for pos in positive_prompts:
                pos_tokens = clip.tokenize(pos)
                pos_encode = clip.encode_from_tokens({"g": pos_tokens["g"], "l": pos_tokens["l"]}, return_pooled=True)
                pos_emb = pos_encode[0][..., 0, 1280:] if isinstance(pos_encode, (tuple, list)) else pos_encode[..., 0, 1280:]
                if pos_emb.ndim == 1:
                    pos_emb = pos_emb.unsqueeze(0)
                positive_embs.append(pos_emb)
            positive_embs = self._stack_embeddings(positive_embs)
            if positive_embs is not None and positive_embs.shape[0] > 0:
                pooled_l = self.addition_vector_add(pooled_l, positive_embs, clip_l_addvec_alpha)
                pooled_l = self.rescale_to_match_base(pooled_l, cond_l)
                cond_l = pooled_l.clone()

        # ----------- CLIP-G Minusvec/Addvec -----------
        if clip_g_minusvec.strip():
            negative_prompts = [s.strip() for s in clip_g_minusvec.split('|') if s.strip()]
            negative_embs = []
            for neg in negative_prompts:
                neg_tokens = clip.tokenize(neg)
                neg_encode = clip.encode_from_tokens({"g": neg_tokens["g"], "l": neg_tokens["l"]}, return_pooled=True)
                neg_emb = neg_encode[0][..., 0, :1280] if isinstance(neg_encode, (tuple, list)) else neg_encode[..., 0, :1280]
                if neg_emb.ndim == 1:
                    neg_emb = neg_emb.unsqueeze(0)
                negative_embs.append(neg_emb)
            negative_embs = self._stack_embeddings(negative_embs)
            if negative_embs is not None and negative_embs.shape[0] > 0:
                if simple_neg == "True":
                    pooled_g = self.vector_subtract_simple(pooled_g, negative_embs, clip_g_minusvec_alpha)
                else:
                    pooled_g = self.vector_subtract(pooled_g, negative_embs, clip_g_minusvec_alpha)
                pooled_g = self.rescale_to_match_base(pooled_g, cond_g)
                cond_g = pooled_g.clone()

        if clip_g_addvec.strip():
            positive_prompts = [s.strip() for s in clip_g_addvec.split('|') if s.strip()]
            positive_embs = []
            for pos in positive_prompts:
                pos_tokens = clip.tokenize(pos)
                pos_encode = clip.encode_from_tokens({"g": pos_tokens["g"], "l": pos_tokens["l"]}, return_pooled=True)
                pos_emb = pos_encode[0][..., 0, :1280] if isinstance(pos_encode, (tuple, list)) else pos_encode[..., 0, :1280]
                if pos_emb.ndim == 1:
                    pos_emb = pos_emb.unsqueeze(0)
                positive_embs.append(pos_emb)
            positive_embs = self._stack_embeddings(positive_embs)
            if positive_embs is not None and positive_embs.shape[0] > 0:
                pooled_g = self.addition_vector_add(pooled_g, positive_embs, clip_g_addvec_alpha)
                pooled_g = self.rescale_to_match_base(pooled_g, cond_g)
                cond_g = pooled_g.clone()

        # Nuke/randn ops as before
        if randn_clip == "True":
            torch.manual_seed(seed)
            cond_l = torch.randn_like(cond_l)
            pooled_l = torch.randn_like(pooled_l)
        if nuke_clip == "True":
            cond_l = torch.zeros_like(cond_l)
            pooled_l = torch.zeros_like(pooled_l)
        if randn_bigclip == "True":
            torch.manual_seed(seed)
            cond_g = torch.randn_like(cond_g)
            pooled_g = torch.randn_like(pooled_g)
        if nuke_bigclip == "True":
            cond_g = torch.zeros_like(cond_g)
            pooled_g = torch.zeros_like(pooled_g)

        cond = torch.cat([cond_g, cond_l], dim=-1)
        pooled = torch.cat([pooled_g, pooled_l], dim=-1)

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


class CLIPTextEncodeFluxDiffVec:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", ),
                "clip_l": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "t5xxl": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "clip_l_minusvec": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "Negative concepts to subtract from CLIP embedding, separated by | (e.g. cat|dog)."}),
                "clip_l_addvec": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "Positive concepts to add from CLIP embedding, separated by | (e.g. cat|dog)."}),
                "t5xxl_minusvec": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "Negative concepts to subtract from T5XXL embedding, separated by | (e.g. cat|dog)."}),
                "t5xxl_addvec": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "Negative concepts to subtract from T5XXL embedding, separated by | (e.g. cat|dog)."}),
                "clip_minusvec_alpha": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step": 0.001, "tooltip": "Scaling for difference vector (subtract mode only)"}),
                "clip_addvec_alpha": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step": 0.001, "tooltip": "Scaling for addition vector (add mode only)"}),
                "t5xxl_minusvec_alpha": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "Scaling for difference vector (subtract mode only)"}),
                "t5xxl_addvec_alpha": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "Scaling for difference vector (subtract mode only)"}),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "simple_neg": (["False", "True"], {"default": "False"}),
                "randn_t5": (["False", "True"], {"default": "False"}),
                "nuke_t5": (["False", "True"], {"default": "False"}),
                "nuke_clip": (["False", "True"], {"default": "False"}),
                "randn_clip": (["False", "True"], {"default": "False"}),
                "seed": ("INT", {"default": 425533035839474, "min": 0, "max": 0xffffffffffffffff}),
                "custom_embeds": (["False", "True"], {"default": "False"}),
                "embeds_idx": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1}),
                "embeds_path": ("STRING", {"default": "path/to/embedding.pt", "multiline": False}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    OUTPUT_TOOLTIPS = ("Nukes or randomizes text encoder input; loads custom embeddings for CLIP guidance; supports difference vectors for both CLIP and T5XXL prompt editing.",)
    FUNCTION = "encode"

    CATEGORY = "zer0int/DiffVec"
    DESCRIPTION = "Nukes or randomizes text encoder input; loads custom embeddings for CLIP guidance; supports difference vectors for CLIP and T5XXL prompt editing."

    @staticmethod
    def vector_subtract(base_emb, negative_embs, alpha=1.0):
        """
        Projects and subtracts subspace (orthogonal projection onto the null space)
        """
        # Handle CLIP: [1, d_model], negatives [num_neg, d_model]
        if negative_embs.dim() == 2 and (base_emb.dim() == 2 or base_emb.dim() == 1):
            # If there are many individual negative embeddings, do PCA first
            if negative_embs.shape[0] >= 5: # Threshold 5 vectors / negative embeddings
                explained_var_cutoff=0.95 # 0.85 - 0.99
                max_pca_rank = (negative_embs.shape[0])-2
                # Center the negatives
                neg_centered = negative_embs - negative_embs.mean(dim=0)
                U, S, Vh = torch.linalg.svd(neg_centered, full_matrices=False)  # Vh: [num_neg, d_model]
                # Select top principal components -> cumulative explained variance
                explained = (S ** 2) / (S ** 2).sum()
                cumexp = explained.cumsum(0)
                k = int((cumexp < explained_var_cutoff).sum().item()) + 1
                k = min(k, max_pca_rank, Vh.shape[0]) # enforce cap
                V = Vh[:k].T  # [d_model, k]
                Q = V  # Q is [d_model, k]
                emb = base_emb.squeeze(0) if base_emb.dim() == 2 else base_emb  # [d_model]
                proj = Q @ (Q.T @ emb)
                result = emb - alpha * proj
                print("[DIFFVEC] ----------- CLIP -----------")
                print(f"[DIFFVEC] PCA projection: used {k} principal vectors out of {negative_embs.shape[0]}")
                print("[DIFFVEC] Norm before:", base_emb.norm().item())
                print("[DIFFVEC] Negs before:", negative_embs.norm().item())
                print("[DIFFVEC] Norm after projection:", proj.norm().item())
                print("[DIFFVEC] Fraction kept:", result.norm().item() / emb.norm().item())
                print("[DIFFVEC] ----------------------------")
                return result.unsqueeze(0) if base_emb.dim() == 2 else result
            else:
                emb = base_emb.squeeze(0) if base_emb.dim() == 2 else base_emb  # [d_model]
                negs = negative_embs  # [num_neg, d_model]
                if negs.shape[0] == 0:
                    return base_emb
                Q, _ = torch.linalg.qr(negs.T, mode='reduced')  # Q: [d_model, num_neg]
                emb_proj = Q @ (Q.T @ emb)
                emb_out = emb - alpha * emb_proj
                print("[DIFFVEC] ----------- CLIP -----------")
                print("[DIFFVEC] Norm before:", emb.norm().item())
                print("[DIFFVEC] Negs before:", negs.norm().item())
                print("[DIFFVEC] Norm after projection:", emb_proj.norm().item())
                print("[DIFFVEC] Fraction kept:", emb_out.norm().item() / emb.norm().item())
                print("[DIFFVEC] ----------------------------")
                return emb_out.unsqueeze(0) if base_emb.dim() == 2 else emb_out

        # Handle T5: [1, seq, d_model] or [seq, d_model], negatives [num_neg, seq, d_model]
        elif negative_embs.dim() == 3:
            emb = base_emb.squeeze(0) if base_emb.dim() == 3 and base_emb.shape[0] == 1 else base_emb  # [seq, d_model]
            num_neg, seq, d_model = negative_embs.shape
            if emb.shape[0] != seq or emb.shape[1] != d_model:
                raise RuntimeError(f"Shape mismatch: base_emb {emb.shape}, negative_embs {negative_embs.shape}")
            out = torch.empty_like(emb)
            for i in range(seq):
                negs = negative_embs[:, i, :]  # [num_neg, d_model]
                v = emb[i]                     # [d_model]
                if negs.shape[0] == 0:
                    out[i] = v
                    continue
                Q, _ = torch.linalg.qr(negs.T, mode='reduced')
                v_proj = Q @ (Q.T @ v)
                out[i] = v - alpha * v_proj          
            print("[DIFFVEC] ----------- T5 -----------")
            print("[DIFFVEC] Norm before:", emb.norm().item())
            print("[DIFFVEC] Negs before:", negative_embs.norm().item())
            print("[DIFFVEC] Norm after:", out.norm().item())
            print("[DIFFVEC] Fraction kept:", out.norm().item() / emb.norm().item())
            print("[DIFFVEC] --------------------------")
            return out.unsqueeze(0) if base_emb.dim() == 3 else out
        else:
            raise RuntimeError(f"Unsupported shapes for vector_subtract: base_emb {base_emb.shape}, negative_embs {negative_embs.shape}")

    @staticmethod
    def vector_subtract_simple(base_emb, negative_embs, alpha=1.0):
        return base_emb - alpha * negative_embs.sum(dim=0, keepdim=True)

    @staticmethod
    def addition_vector_add(base_emb, positive_embs, alpha=1.0):
        # Elementwise addition of one or more positive concept embeddings, scaled by alpha
        return base_emb + alpha * positive_embs.sum(dim=0, keepdim=True)

    @staticmethod
    def rescale_to_match_base(final_emb, base_emb):
        """
        Scales final_emb so that its L2 norm matches that of base_emb.
        Both tensors should be [1, D] or [B, D].
        """
        final_norm = final_emb.norm(dim=-1, keepdim=True)
        base_norm = base_emb.norm(dim=-1, keepdim=True)
        scale = (base_norm / (final_norm + 1e-6))
        print("[DIFFVEC] ---- Rescale L2 Norm ----")
        print("[DIFFVEC] Base embed:", base_emb.norm().item())
        print("[DIFFVEC] Final embed:", final_emb.norm().item())
        return final_emb * scale

    def encode(self, clip, clip_l, t5xxl, clip_l_minusvec, clip_l_addvec, t5xxl_minusvec, t5xxl_addvec, clip_minusvec_alpha, clip_addvec_alpha, t5xxl_minusvec_alpha, t5xxl_addvec_alpha, guidance, simple_neg, randn_t5, nuke_t5, nuke_clip, randn_clip, custom_embeds, embeds_path, embeds_idx, seed):
        # Custom Embeddings Branch (load file):
        if custom_embeds == "True":
            tokens = clip.tokenize(clip_l)
            tokens["t5xxl"] = clip.tokenize(t5xxl)["t5xxl"]
            output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
            clip_embeddings = output["pooled_output"]
            original_clip_embedding = clip_embeddings.clone()
            modified_clip_embeddings = clip_embeddings

            if not os.path.exists(embeds_path):
                raise FileNotFoundError(f"Custom embedding file not found: {embeds_path}")
            custom_embedding = torch.load(embeds_path, map_location="cpu").float()
            if custom_embedding.dim() != 2 or custom_embedding.shape[1] != 768:
                raise ValueError(f"Custom embedding must have shape [batch_size, 768]. Found: {custom_embedding.shape}")
            num_embeddings = custom_embedding.size(0)
            if embeds_idx >= num_embeddings or embeds_idx < 0:
                embeds_idx = 0
            selected_embedding = custom_embedding[embeds_idx:embeds_idx + 1].float()
            modified_clip_embeddings = selected_embedding
            output["pooled_output"] = modified_clip_embeddings
            original_clip_embedding = modified_clip_embeddings.clone()  # Reset reference after custom load

            # --- CLIP minusvec (use pooled_output, only CLIP tokens) ---
            if clip_l_minusvec.strip():
                if simple_neg == "True":     
                    negative_prompts = [s.strip() for s in clip_l_minusvec.split('|') if s.strip()]
                    if negative_prompts:
                        negative_embs = []
                        for neg in negative_prompts:
                            neg_emb = clip.encode_from_tokens(clip.tokenize(neg), return_pooled=True, return_dict=True)["pooled_output"]
                            negative_embs.append(neg_emb.squeeze(0))
                        negative_embs = torch.stack(negative_embs, dim=0)
                        modified_clip_embeddings = self.vector_subtract_simple(modified_clip_embeddings, negative_embs, clip_minusvec_alpha)
                        modified_clip_embeddings = self.rescale_to_match_base(modified_clip_embeddings, original_clip_embedding)
                        output["pooled_output"] = modified_clip_embeddings
                else:
                    negative_prompts = [s.strip() for s in clip_l_minusvec.split('|') if s.strip()]
                    if negative_prompts:
                        negative_embs = []
                        for neg in negative_prompts:
                            neg_emb = clip.encode_from_tokens(clip.tokenize(neg), return_pooled=True, return_dict=True)["pooled_output"]
                            negative_embs.append(neg_emb.squeeze(0))
                        negative_embs = torch.stack(negative_embs, dim=0)
                        modified_clip_embeddings = self.vector_subtract(modified_clip_embeddings, negative_embs, clip_minusvec_alpha)
                        modified_clip_embeddings = self.rescale_to_match_base(modified_clip_embeddings, original_clip_embedding)
                        output["pooled_output"] = modified_clip_embeddings

            # --- CLIP addvec (use pooled_output, only CLIP tokens) ---
            if clip_l_addvec.strip():
                positive_prompts = [s.strip() for s in clip_l_addvec.split('|') if s.strip()]
                if positive_prompts:
                    positive_embs = []
                    for pos in positive_prompts:
                        pos_emb = clip.encode_from_tokens(clip.tokenize(pos), return_pooled=True, return_dict=True)["pooled_output"]
                        positive_embs.append(pos_emb.squeeze(0))
                    positive_embs = torch.stack(positive_embs, dim=0)                    
                    base_emb = self.addition_vector_add(output["pooled_output"], positive_embs, clip_addvec_alpha)                    
                    base_emb = self.rescale_to_match_base(base_emb, original_clip_embedding)
                    output["pooled_output"] = base_emb

            # --- T5XXL minusvec (dodgy thing to do with a T5, but can tune / reduce influence of T5) ---
            if t5xxl_minusvec.strip():
                main_t5_token = clip.tokenize(t5xxl)["t5xxl"]
                t5_cond = clip.t5xxl(main_t5_token)

                negative_prompts = [s.strip() for s in t5xxl_minusvec.split('|') if s.strip()]
                if negative_prompts:
                    negative_embs = []
                    for neg in negative_prompts:
                        neg_t5_token = clip.tokenize(neg)["t5xxl"]
                        neg_emb = clip.t5xxl(neg_t5_token)
                        negative_embs.append(neg_emb.squeeze(0))  # [seq, d_model]
                    negative_embs = torch.stack(negative_embs, dim=0)  # [num_neg, seq, d_model]
                    t5_cond = t5_cond - t5xxl_minusvec_alpha * negative_embs.mean(dim=0, keepdim=True)
                t5_cond = self.rescale_to_match_base(t5_cond, output["cond"])
                output["cond"] = t5_cond

            # --- T5XXL addvec (semantic negative, vector positive -> e.g. add "no flowers") ---
            if t5xxl_addvec.strip():
                positive_prompts = [s.strip() for s in t5xxl_addvec.split('|') if s.strip()]
                if positive_prompts:
                    positive_embs = []
                    for pos in positive_prompts:
                        pos_t5_token = clip.tokenize(pos)["t5xxl"]
                        tokens = {"t5xxl": pos_t5_token}
                        pos_emb = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)["cond"]
                        if pos_emb.dim() == 3 and pos_emb.shape[0] == 1:
                            pos_emb = pos_emb.squeeze(0)
                        positive_embs.append(pos_emb)
                    positive_embs = torch.stack(positive_embs, dim=0)
                    t5_cond = output["cond"]
                    if t5_cond.dim() == 3 and t5_cond.shape[0] == 1:
                        t5_cond = t5_cond.squeeze(0)
                    t5_cond = t5_cond + t5xxl_addvec_alpha * positive_embs.sum(dim=0)#mean
                    t5_cond = self.rescale_to_match_base(t5_cond, output["cond"])
                    output["cond"] = t5_cond

            # ------- Zero or Random the Text Encoder input to the Diffusion Model -------
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
        
        # ----------------------------------------------- BRANCH SPLIT -----------------------------------------------
        
        # Embedding from 'use prompt' branch:
        else:
            tokens = clip.tokenize(clip_l)
            tokens["t5xxl"] = clip.tokenize(t5xxl)["t5xxl"]
            output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
            original_clip_embedding = output["pooled_output"].clone()
            base_emb = output["pooled_output"]

            # --- CLIP minusvec (use pooled_output, only CLIP tokens) ---
            if clip_l_minusvec.strip():
                if simple_neg == "True":
                    negative_prompts = [s.strip() for s in clip_l_minusvec.split('|') if s.strip()]
                    if negative_prompts:
                        negative_embs = []
                        for neg in negative_prompts:
                            neg_emb = clip.encode_from_tokens(clip.tokenize(neg), return_pooled=True, return_dict=True)["pooled_output"]
                            negative_embs.append(neg_emb.squeeze(0))
                        negative_embs = torch.stack(negative_embs, dim=0)
                        base_emb = self.vector_subtract_simple(base_emb, negative_embs, clip_minusvec_alpha)
                        base_emb = self.rescale_to_match_base(base_emb, original_clip_embedding)
                        output["pooled_output"] = base_emb                
                else:
                    negative_prompts = [s.strip() for s in clip_l_minusvec.split('|') if s.strip()]
                    if negative_prompts:
                        negative_embs = []
                        for neg in negative_prompts:
                            neg_emb = clip.encode_from_tokens(clip.tokenize(neg), return_pooled=True, return_dict=True)["pooled_output"]
                            negative_embs.append(neg_emb.squeeze(0))
                        negative_embs = torch.stack(negative_embs, dim=0)
                        base_emb = self.vector_subtract(base_emb, negative_embs, clip_minusvec_alpha)
                        base_emb = self.rescale_to_match_base(base_emb, original_clip_embedding)
                        output["pooled_output"] = base_emb

            # --- CLIP addvec (positive concepts) ---
            if clip_l_addvec.strip():
                positive_prompts = [s.strip() for s in clip_l_addvec.split('|') if s.strip()]
                if positive_prompts:
                    positive_embs = []
                    for pos in positive_prompts:
                        pos_emb = clip.encode_from_tokens(clip.tokenize(pos), return_pooled=True, return_dict=True)["pooled_output"]
                        positive_embs.append(pos_emb.squeeze(0))
                    positive_embs = torch.stack(positive_embs, dim=0)
                    base_emb = self.addition_vector_add(output["pooled_output"], positive_embs, clip_addvec_alpha)   
                    base_emb = self.rescale_to_match_base(base_emb, original_clip_embedding)
                    output["pooled_output"] = base_emb


            # --- T5XXL minusvec (must provide both keys) ---
            if t5xxl_minusvec.strip() and "cond" in output and output["cond"] is not None:
                main_l_token = clip.tokenize(clip_l)["l"]
                negative_prompts = [s.strip() for s in t5xxl_minusvec.split('|') if s.strip()]
                if negative_prompts:
                    negative_embs = []
                    for neg in negative_prompts:
                        t5_token = clip.tokenize(neg)["t5xxl"]
                        tokens_dict = {"l": main_l_token, "t5xxl": t5_token}
                        neg_emb = clip.encode_from_tokens(tokens_dict, return_pooled=True, return_dict=True)["cond"]
                        negative_embs.append(neg_emb.squeeze(0))
                    negative_embs = torch.stack(negative_embs, dim=0)
                    base_emb = output["cond"]
                    original_embed = output["cond"]
                    base_emb = self.vector_subtract(base_emb, negative_embs, t5xxl_minusvec_alpha)
                    output["cond"] = self.rescale_to_match_base(base_emb, original_embed)

            # --- T5XXL addvec (positive concepts) ---
            if t5xxl_addvec.strip():
                positive_prompts = [s.strip() for s in t5xxl_addvec.split('|') if s.strip()]
                if positive_prompts:
                    positive_embs = []
                    # Cache the main CLIP token once for 'l'
                    main_clip_token = clip.tokenize(clip_l)["l"]
                    for pos in positive_prompts:
                        pos_t5_token = clip.tokenize(pos)["t5xxl"]
                        tokens = {"l": main_clip_token, "t5xxl": pos_t5_token}
                        pos_emb = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)["cond"]
                        if pos_emb.dim() == 3 and pos_emb.shape[0] == 1:
                            pos_emb = pos_emb.squeeze(0)   # [seq, d_model]
                        positive_embs.append(pos_emb)
                    positive_embs = torch.stack(positive_embs, dim=0)      # [num_add, seq, d_model]
                    t5_cond = output["cond"]
                    if t5_cond.dim() == 3 and t5_cond.shape[0] == 1:
                        t5_cond = t5_cond.squeeze(0)      # [seq, d_model]
                    # Addition: sum or mean
                    t5_cond = t5_cond + t5xxl_addvec_alpha * positive_embs.mean(dim=0)  # or .mean(dim=0)
                    t5_cond = self.rescale_to_match_base(t5_cond, output["cond"])
                    # Restore batch dimension for output
                    if t5_cond.dim() == 2:
                        t5_cond = t5_cond.unsqueeze(0)    # [1, seq, d_model]
                    output["cond"] = t5_cond

            # ------- Zero or Random the Text Encoder input to the Diffusion Model -------
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