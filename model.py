from __future__ import annotations

import os

import pretrainedmodels
import torch
import torch.nn as nn
from torchvision import transforms


class XceptionDetector:
    DEFAULT_MODEL_PATH = r"E:\Data\study_code\py_code\Graph_FakeDetector\I3D_8x8_R50.pth"

    def __init__(self, model_path=None, device: str = "cuda"):
        self.device = device

        model = pretrainedmodels.xception(pretrained="imagenet")
        model.last_linear = nn.Linear(model.last_linear.in_features, 2)

        path_to_load = model_path if model_path else self.DEFAULT_MODEL_PATH
        if path_to_load and os.path.exists(path_to_load):
            model.load_state_dict(torch.load(path_to_load, map_location=device))
            print(f"[OK] Loaded detector weights from `{path_to_load}`")
        else:
            if model_path:
                print(
                    f"[WARN] Weight file `{path_to_load}` not found. "
                    "Using ImageNet initialization only."
                )
            else:
                print(
                    "[WARN] Custom detector weights not found. "
                    "Using ImageNet initialization only; deepfake quality will be limited."
                )

        self.model = model.eval().to(device)
        self.transform = transforms.Compose(
            [
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
            ]
        )

    def predict(self, images):
        """
        Run inference for one PIL image or a list of PIL images.

        Returns a dict for single input or a list[dict] for batch input.
        """
        if isinstance(images, list):
            is_batch = True
            image_list = images
        else:
            is_batch = False
            image_list = [images]

        tensor_list = []
        for image in image_list:
            if image.mode != "RGB":
                image = image.convert("RGB")
            tensor_list.append(self.transform(image))

        batch_tensor = torch.stack(tensor_list, dim=0).to(self.device)

        with torch.no_grad():
            output = self.model(batch_tensor)
            probs = torch.softmax(output, dim=1)

        results = []
        for index in range(probs.size(0)):
            fake_prob = probs[index, 1].item()
            real_prob = probs[index, 0].item()
            is_fake = fake_prob > 0.5
            results.append(
                {
                    "is_fake": bool(is_fake),
                    "label": "fake" if is_fake else "real",
                    "confidence": round(max(fake_prob, real_prob), 4),
                    "fake_score": round(fake_prob, 4),
                }
            )

        return results if is_batch else results[0]
