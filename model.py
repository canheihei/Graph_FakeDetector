from flask import Flask, request, jsonify
from PIL import Image
import io
import torch
import torch.nn as nn
from torchvision import transforms
import pretrainedmodels



class XceptionDetector:
    DEFAULT_MODEL_PATH = r"E:\Data\study_code\py_code\Graph_FakeDetector\xxI3D_8x8_R50.pth"

    def __init__(self, model_path=None, device="cpu"):
        import os
        self.device = device

        model = pretrainedmodels.xception(pretrained='imagenet')
        model.last_linear = nn.Linear(model.last_linear.in_features, 2)

        # 使用传入路径或默认路径
        path_to_load = model_path if model_path else self.DEFAULT_MODEL_PATH

        if path_to_load and os.path.exists(path_to_load):
            model.load_state_dict(torch.load(path_to_load, map_location=device))
            print(f"已加载微调权重：`{path_to_load}`")
        else:
            if model_path:
                print(f"权重文件 `{path_to_load}` 未找到，使用 ImageNet 预训练权重作为回退")
            else:
                print("未提供微调权重，使用 ImageNet 预训练权重（精度可能不合适于 deepfake 检测）")

        self.model = model.eval().to(device)

        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def predict(self, image: Image.Image):
        img = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(img)
            prob = torch.softmax(output, dim=1)[0]
            fake_prob = prob[1].item()
            real_prob = prob[0].item()

        is_fake = fake_prob > 0.5
        return {
            "is_fake": bool(is_fake),
            "label": "fake" if is_fake else "real",
            "confidence": round(max(fake_prob, real_prob), 4),
            "fake_score": round(fake_prob, 4)
        }