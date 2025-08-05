from . import diffvec as diffvec

NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodeFluxDiffVec": diffvec.CLIPTextEncodeFluxDiffVec,
    "CLIPTextEncodeDiffVec": diffvec.CLIPTextEncodeDiffVec,
    "CLIPTextEncodeSDXLDiffVec": diffvec.CLIPTextEncodeSDXLDiffVec,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTextEncodeFluxDiffVec": "CLIP-FLUX DiffVec",
    "CLIPTextEncodeDiffVec": "CLIP-SD1 DiffVec",
    "CLIPTextEncodeSDXLDiffVec": "CLIP-SDXL DiffVec",
}