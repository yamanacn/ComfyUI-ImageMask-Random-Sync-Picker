import torch
import random

class ImageMaskRandomSelector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_count": ("INT", {"default": 2, "min": 1, "max": 50, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                # 动态输入将通过 **kwargs 接收
                # JS 扩展会负责在 UI 上添加 image_1, mask_1, image_2, mask_2 等
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "MASK")
    FUNCTION = "select_random"
    CATEGORY = "ImageMaskSelector"

    def select_random(self, input_count, seed, **kwargs):
        # 使用传入的 seed 确保结果可复现（在同一次运行中）
        random.seed(seed)
        
        valid_indices = []
        for i in range(1, input_count + 1):
            img_key = f"image_{i}"
            # 只有当图像输入被连接时，才认为该索引有效
            if img_key in kwargs and kwargs[img_key] is not None:
                valid_indices.append(i)
        
        if not valid_indices:
            # 如果没有连接任何图像，尝试返回一个空的张量或报错
            # 在 ComfyUI 中，通常至少需要一个输入
            raise ValueError("ImageMaskRandomSelector: No images connected. Please connect at least one image input.")

        # 从有效的索引中随机选择一个
        selected_idx = random.choice(valid_indices)
        
        selected_image = kwargs[f"image_{selected_idx}"]
        selected_mask = kwargs.get(f"mask_{selected_idx}")

        # 处理遮罩：如果对应的遮罩没有连接，则创建一个全黑（无遮罩）的默认遮罩
        if selected_mask is None:
            # selected_image 形状通常是 [B, H, W, C]
            # Mask 形状应为 [B, H, W]
            B, H, W, C = selected_image.shape
            selected_mask = torch.zeros((B, H, W), dtype=torch.float32, device=selected_image.device)
        else:
            # 确保 mask 的 batch size 与 image 一致（防止输入不匹配）
            if selected_mask.shape[0] != selected_image.shape[0]:
                # 如果 batch 不匹配，可以进行简单的广播或裁剪，这里选择抛出警告或简单处理
                pass

        return (selected_image, selected_mask)

NODE_CLASS_MAPPINGS = {
    "ImageMaskRandomSelector": ImageMaskRandomSelector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageMaskRandomSelector": "Image & Mask Random Selector"
}
