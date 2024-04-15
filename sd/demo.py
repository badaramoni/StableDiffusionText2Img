import model_loader
import pipeline
from PIL import Image
import torch
from transformers import CLIPTokenizer
import os


DEVICE = "cpu"
ALLOW_CUDA = True
ALLOW_MPS = True

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device: {DEVICE}")


tokenizer = CLIPTokenizer("data/vocab.json", merges_file="data/merges.txt")
model_file = "data/v1-5-pruned-emaonly.ckpt"
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)


prompt = "a dog in a park"
uncond_prompt = ""
do_cfg = True
cfg_scale = 8


sampler = "ddpm"
num_inference_steps = 50
seed = 42

# Generate image
output_image = pipeline.generate(
    prompt=prompt,
    uncond_prompt=uncond_prompt,
    input_image=None,
    strength=0.9,
    do_cfg=do_cfg,
    cfg_scale=cfg_scale,
    sampler_name=sampler,
    n_inference_steps=num_inference_steps,
    seed=seed,
    models=models,
    device=DEVICE,
    idle_device="cpu",
    tokenizer=tokenizer,
)

# Define the path to the Downloads folder
downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads')
output_image_path = os.path.join(downloads_path, 'generated_image.png')

# Save the output image
image = Image.fromarray(output_image)
image.save(output_image_path)
print(f"Image saved to {output_image_path}")
