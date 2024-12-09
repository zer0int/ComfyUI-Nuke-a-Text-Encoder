### ComfyUI Nukes & Embeddings for Text Encoders! 🤯

- Put the "ComfyUI-Nuke-a-TE" folder into "ComfyUI/custom_nodes" and run Comfy.
- Nuke a text encoder (zero the image-guiding input)!
- Nuke T5 to guide Flux.1-dev with CLIP only!
- (Make AI crazy again! 🤪)
- Use a random distribution (torch.randn) for CLIP *and* T5! 🥳
- Explore Flux.1's bias as it stares into itelf! 👀
- Nodes (and workflows) for Flux.1, SD-1.5, SDXL ✅
- You should totally [let CLIP rant about your image](https://github.com/zer0int/CLIP-gradient-ascent-embeddings) and use that embedding to steer an image. 🤗
- You should totally [download my new CLIP-SAE Text Encoder](https://huggingface.co/zer0int/CLIP-SAE-ViT-L-14) from HuggingFace! 🤗
- Node for SDXL is work of art by OpenAI's o1; the AI decided to slice the tensor and inject the CLIP embedding as it couldn't find where to separate it. This is very wrong, and very right, and totally an AI's art, and so be it! Slice those tensors and make it more... madness! 🤖
------
o1 did this (SDXL).

![ai-is-artist](https://github.com/user-attachments/assets/abb1fbd4-6b90-4904-b5d8-17266e2ac38b)
------
CLIP still can't write coherent text, and never will. But it makes fun things on its own! (Flux.1-dev)

![cant-draw](https://github.com/user-attachments/assets/3a33d1e3-0c50-4b3d-8106-04ea99fd7abf)
------
My new CLIP fine-tune, linked above! (CLIP only Flux.1 guidance, 🚫no T5💥

![mathematcloseup-1](https://github.com/user-attachments/assets/27d13bdd-6553-4d30-a62b-051f1294fbdf)
------
A bit of love & strangeness for good old SD-1.5 🙃

![stable confusion](https://github.com/user-attachments/assets/14319211-66c6-4256-9a12-474824bfad5d)
