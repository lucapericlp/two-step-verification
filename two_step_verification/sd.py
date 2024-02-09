from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image
import cv2
import torch
import numpy as np
from PIL import Image

from two_step_verification.mps_backend import has_mps


class TextDreamer:
    def __init__(self):
        self.pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
        if has_mps():
            self.pipe.to("mps")

    def __call__(self, init_image: np.ndarray) -> np.ndarray:
        prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."

        return np.array(
            self.pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
        )

class Dreamer:
    def __init__(self):
        self.pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo")
        if has_mps():
            self.pipe.to("mps")

    def __call__(self, init_image: np.ndarray) -> np.ndarray:
        _image = cv2.resize(init_image, (512, 512))
        prompt = "anime, portrait, girl with black long hair and glasses, big smile"
        image = Image.fromarray(_image)
        image.save("test.jpg")

        return np.array(
            self.pipe(
                prompt,
                image=image,
                num_inference_steps=2,
                strength=0.5,
                guidance_scale=0.0
            ).images[0]
        )
