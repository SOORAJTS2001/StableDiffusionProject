import datetime

from diffusers import DiffusionPipeline
import numpy as np
from pathlib import Path
from upscalers.updated_upscaler import upscale

generated_image_path = str(Path('generated_image').resolve())
upscaled_image_path = str(Path('upscaled_image').resolve())
model_path = str(Path('models').resolve()) + '/RRDB_ESRGAN_x4.pth'
prompt = "astronaut riding a horse"


def generateImage(prompt, generated_image_path, upscaled_image_path, model_path):
    pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    # pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
    pipe = pipe.to("mps")
    # pipe.enable_model_cpu_offload()
    # Recommended if your computer has < 64 GB of RAM
    pipe.enable_attention_slicing()
    # First-time "warmup" pass if PyTorch version is 1.13 (see explanation above)
    _ = pipe(prompt, num_inference_steps=1)
    # Results match those from the CPU device after the warmup pass.
    image = pipe(prompt).images[0]

    image_array = np.array(image)
    # timestamp_name = datetime.datetime.now().strftime("%Y-%m-%d|%H:%M:%S")+".png"
    # image.save(generated_image_path+"/"+timestamp_name)
    upscaled_image_path,timestamp_name = upscale(model_path, image_array, upscaled_image_path)
    return upscaled_image_path,timestamp_name



if __name__ == "__main__":
    generateImage(prompt, generated_image_path, upscaled_image_path, model_path)
