from . import nukete as nukete

NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodeFluxNUKE": nukete.CLIPTextEncodeFluxNUKE,
    "CLIPTextEncodeNUKE": nukete.CLIPTextEncodeNUKE,
    "CLIPTextEncodeSDXLNUKE": nukete.CLIPTextEncodeSDXLNUKE,
    "CLIPTextEncodeSDXLminiNUKE": nukete.CLIPTextEncodeSDXLminiNUKE,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTextEncodeFluxNUKE": "CLIP-FLUX NUKE",
    "CLIPTextEncodeNUKE": "CLIP-SD1 NUKE",
    "CLIPTextEncodeSDXLNUKE": "CLIP-SDXL NUKE",
    "CLIPTextEncodeSDXLminiNUKE": "CLIP-SDXL negative",
}