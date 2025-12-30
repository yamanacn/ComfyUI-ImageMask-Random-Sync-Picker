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
        # 使用传入的 seed 确保结果可复现
        random.seed(seed)
        
        valid_indices = []
        for i in range(1, input_count + 1):
            img_key = f"image_{i}"
            # 只有当图像输入被连接时，才认为该索引有效
            if img_key in kwargs and kwargs[img_key] is not None:
                valid_indices.append(i)
        
        if not valid_indices:
            raise ValueError("ImageMaskRandomSelector: No images connected. Please connect at least one image input.")

        # 从有效的索引中随机选择一个
        selected_idx = random.choice(valid_indices)
        
        selected_image = kwargs.get(f"image_{selected_idx}")
        selected_mask = kwargs.get(f"mask_{selected_idx}")

        print(f"ImageMaskRandomSelector: Selected index {selected_idx}")

        # 确保 image 是 tensor 且不为 None
        if selected_image is None or not isinstance(selected_image, torch.Tensor):
            print(f"ImageMaskRandomSelector: selected_image is {type(selected_image)}, trying to find fallback...")
            for idx in valid_indices:
                temp_img = kwargs.get(f"image_{idx}")
                if temp_img is not None and isinstance(temp_img, torch.Tensor):
                    selected_image = temp_img
                    selected_idx = idx
                    selected_mask = kwargs.get(f"mask_{idx}")
                    print(f"ImageMaskRandomSelector: Fallback to index {idx}")
                    break
            
            if selected_image is None or not isinstance(selected_image, torch.Tensor):
                raise TypeError("ImageMaskRandomSelector: Could not find a valid image tensor.")

        # 彻底确保 mask 永远不是 None
        if selected_mask is None:
            print(f"ImageMaskRandomSelector: mask_{selected_idx} is None, creating default zero mask.")
            # 获取图像的形状来创建默认遮罩 [B, H, W, C] -> [B, H, W]
            B, H, W = selected_image.shape[0], selected_image.shape[1], selected_image.shape[2]
            selected_mask = torch.zeros((B, H, W), dtype=torch.float32, device=selected_image.device)
        elif not isinstance(selected_mask, torch.Tensor):
            print(f"ImageMaskRandomSelector: mask_{selected_idx} is not a tensor ({type(selected_mask)}), converting or creating default.")
            try:
                selected_mask = torch.from_numpy(selected_mask) if hasattr(selected_mask, "numpy") else torch.tensor(selected_mask)
            except:
                B, H, W = selected_image.shape[0], selected_image.shape[1], selected_image.shape[2]
                selected_mask = torch.zeros((B, H, W), dtype=torch.float32, device=selected_image.device)

        # 检查并修复 mask 维度，确保它是 [B, H, W]
        if hasattr(selected_mask, "dim"):
            if selected_mask.dim() == 2:
                selected_mask = selected_mask.unsqueeze(0)
            elif selected_mask.dim() == 4:
                # 如果是 [B, H, W, C]，取第一个通道
                selected_mask = selected_mask[:, :, :, 0]
        
        # 确保 batch size 与 image 一致
        if selected_mask.shape[0] != selected_image.shape[0]:
            if selected_mask.shape[0] == 1:
                # 如果 mask 只有一个 batch，repeat 到 image 的 batch 大小
                selected_mask = selected_mask.repeat(selected_image.shape[0], 1, 1)
            else:
                # 如果 batch 不匹配且无法简单 repeat，则根据 image 重新创建一个空 mask
                B, H, W = selected_image.shape[0], selected_image.shape[1], selected_image.shape[2]
                selected_mask = torch.zeros((B, H, W), dtype=torch.float32, device=selected_image.device)

        # 最后的终极检查，绝对不允许返回 None
        if selected_mask is None:
             B, H, W = selected_image.shape[0], selected_image.shape[1], selected_image.shape[2]
             selected_mask = torch.zeros((B, H, W), dtype=torch.float32, device=selected_image.device)

        print(f"ImageMaskRandomSelector: Returning IMAGE {selected_image.shape}, MASK {selected_mask.shape}")
        return (selected_image, selected_mask)

NODE_CLASS_MAPPINGS = {
    "ImageMaskRandomSelector": ImageMaskRandomSelector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageMaskRandomSelector": "Image & Mask Random Selector"
}
