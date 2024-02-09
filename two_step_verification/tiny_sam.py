import torch
import numpy as np
from tinysam import sam_model_registry, SamHierarchicalMaskGenerator
from tinysam.predictor import SamPredictor
from two_step_verification.mps_backend import has_mps

class TinySamEverything:
    def __init__(self):
        print("Setting up TinySam...")
        model_type = "vit_t"
        sam = sam_model_registry[model_type](checkpoint="./weights/tinysam.pth")
        device = "mps" if has_mps() else "cpu"
        sam.to(device=device)
        sam.eval()
        self.pipe = SamHierarchicalMaskGenerator(sam)
        print("Finished setting up TinySam...")

    def __call__(self, image: np.ndarray) -> np.ndarray:
        print("Computing...")
        masks = self.pipe.hierarchical_generate(image)
        sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        print("Computed!")
        return img

class TinySamPoint:
    def __init__(self):
        print("Setting up TinySamPoint...")
        model_type = "vit_t"
        sam = sam_model_registry[model_type](checkpoint="./weights/tinysam.pth")
        device = "mps" if has_mps() else "cpu"
        sam.to(device=device)
        sam.eval()
        self.pipe = SamPredictor(sam)
        print("Finished setting up TinySamPoint...")

    def __call__(self, image: np.ndarray) -> np.ndarray:
        print("Computing...")
        self.pipe.set_image(image)
        use_box = True
        input_box = np.array([640, 360, 1280, 720])
        input_point = np.array([[400, 400]])
        input_label = np.array([1])
        if not use_box:
            masks, scores, logits = self.pipe.predict(
                point_coords=input_point,
                point_labels=input_label,
            )
        else:
            masks, scores, logits = self.pipe.predict(box=input_box)

        all_masks = True
        img = np.zeros((image.shape[0], image.shape[1], 4))
        img[:,:,3] = 0
        if all_masks:
            for mask in masks:
                color_mask = np.concatenate([np.random.random(3), [0.35]])
                img[mask] = color_mask
        else:
            mask = masks[scores.argmax(),:,:]
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[mask] = color_mask
        print("Computed!")
        return img
