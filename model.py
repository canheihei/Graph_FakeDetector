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

    def predict(self, images):
        """
        支持单张或批量图像输入。
        :param images: PIL.Image.Image or List[PIL.Image.Image]
        :return: dict (single) or List[dict] (batch)
        """
        # 判断输入是否为列表
        if isinstance(images, list):
            is_batch = True
            image_list = images
        else:
            is_batch = False
            image_list = [images]

        # 预处理所有图像
        tensor_list = []
        for img in image_list:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            tensor_list.append(self.transform(img))

        # 拼成 batch tensor
        batch_tensor = torch.stack(tensor_list, dim=0).to(self.device)

        # 推理
        with torch.no_grad():
            output = self.model(batch_tensor)
            probs = torch.softmax(output, dim=1)  # [B, 2]

        # 解析结果
        results = []
        for i in range(probs.size(0)):
            fake_prob = probs[i, 1].item()
            real_prob = probs[i, 0].item()
            is_fake = fake_prob > 0.5
            results.append({
                "is_fake": bool(is_fake),
                "label": "fake" if is_fake else "real",
                "confidence": round(max(fake_prob, real_prob), 4),
                "fake_score": round(fake_prob, 4)
            })

        # 返回单个 dict 或 list
        return results if is_batch else results[0]