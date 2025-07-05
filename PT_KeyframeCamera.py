import torch
import torch.nn.functional as F

MAX_RESOLUTION = 8192

class PT_KeyframeCamera:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 9999999, "step": 1}),
                "end_frame": ("INT", {"default": 24, "min": 0, "max": 9999999, "step": 1}),
                # 允许负值位移的关键修改
                "start_horizontal_shift": ("INT", {"default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1, 
                                                "tooltip": "Horizontal shift at start frame (positive=right, negative=left)"}),
                "start_vertical_shift": ("INT", {"default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1,
                                              "tooltip": "Vertical shift at start frame (positive=down, negative=up)"}),
                "start_zoom": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "end_horizontal_shift": ("INT", {"default": 100, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1,
                                               "tooltip": "Horizontal shift at end frame (positive=right, negative=left)"}),
                "end_vertical_shift": ("INT", {"default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1,
                                             "tooltip": "Vertical shift at end frame (positive=down, negative=up)"}),
                "end_zoom": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "zoom_origin": (["center", "top-left", "top-right", "bottom-left", "bottom-right"], {"default": "center"}),
                "pad_mode": (["color", "edge"], {"default": "color"}),
                "bg_color": ("STRING", {"default": "0, 0, 0", "tooltip": "RGB values (0-255) separated by commas"}),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "masks")
    FUNCTION = "apply_camera_movement"
    CATEGORY = "KJNodes/animation"
    DESCRIPTION = "Keyframe-based camera movement (pan & zoom) for image sequences"

    def apply_camera_movement(self, images, start_frame, end_frame, 
                             start_horizontal_shift, start_vertical_shift, start_zoom,
                             end_horizontal_shift, end_vertical_shift, end_zoom,
                             zoom_origin, pad_mode, bg_color):
        
        # 确保输入是图像序列
        if len(images.shape) != 4:
            raise ValueError("Input must be a 4D tensor [batch, height, width, channels]")
        
        num_frames, H, W, C = images.shape
        
        # 解析背景颜色
        try:
            bg_rgb = [float(x.strip()) / 255.0 for x in bg_color.split(",")]
            if len(bg_rgb) == 1:  # 灰度转RGB
                bg_rgb = bg_rgb * 3
            elif len(bg_rgb) != 3:
                raise ValueError("Invalid color format")
        except:
            bg_rgb = [0, 0, 0]  # 默认为黑色
        
        # 转换为tensor
        bg_tensor = torch.tensor(bg_rgb, dtype=images.dtype, device=images.device).view(1, 1, 1, 3)
        
        # 创建输出序列
        output_images = []
        output_masks = []
        
        # 处理序列中的每一帧
        for frame_idx in range(num_frames):
            # 计算当前绝对帧号
            current_frame = frame_idx
            
            # 计算动画进度 (0到1)
            if current_frame < start_frame:
                progress = 0.0
            elif current_frame > end_frame:
                progress = 1.0
            else:
                total_frames = max(1, (end_frame - start_frame))
                progress = (current_frame - start_frame) / total_frames
            
            # 插值计算当前帧的位移和缩放
            current_h_shift = int(round(start_horizontal_shift + (end_horizontal_shift - start_horizontal_shift) * progress))
            current_v_shift = int(round(start_vertical_shift + (end_vertical_shift - start_vertical_shift) * progress))
            current_zoom = start_zoom + (end_zoom - start_zoom) * progress
            
            # 计算缩放后的尺寸
            new_H = max(1, int(H * current_zoom))
            new_W = max(1, int(W * current_zoom))
            
            # 缩放当前帧
            img = images[frame_idx].unsqueeze(0).permute(0, 3, 1, 2)  # [1, C, H, W]
            scaled = F.interpolate(img, size=(new_H, new_W), mode='bilinear', align_corners=False)
            scaled_tensor = scaled.permute(0, 2, 3, 1)  # [1, new_H, new_W, C]
            
            # 计算缩放后的偏移
            origin_offset_x, origin_offset_y = 0, 0
            if zoom_origin == "center":
                origin_offset_x = int((W - new_W) // 2)
                origin_offset_y = int((H - new_H) // 2)
            elif zoom_origin == "top-right":
                origin_offset_x = int(W - new_W)
            elif zoom_origin == "bottom-left":
                origin_offset_y = int(H - new_H)
            elif zoom_origin == "bottom-right":
                origin_offset_x = int(W - new_W)
                origin_offset_y = int(H - new_H)
            
            # 计算最终偏移（支持负值位移）
            final_offset_x = origin_offset_x + current_h_shift
            final_offset_y = origin_offset_y + current_v_shift
            
            # 创建输出画布
            canvas = bg_tensor.expand(1, H, W, 3).clone()
            mask = torch.ones((1, H, W), dtype=torch.float32, device=images.device)
            
            # 计算粘贴区域（支持负值位移）
            paste_x1 = int(max(0, final_offset_x))
            paste_y1 = int(max(0, final_offset_y))
            paste_x2 = int(min(W, final_offset_x + new_W))
            paste_y2 = int(min(H, final_offset_y + new_H))
            
            # 计算源图像裁剪区域（支持负值位移）
            src_x1 = int(max(0, -final_offset_x))
            src_y1 = int(max(0, -final_offset_y))
            src_x2 = int(min(new_W, W - final_offset_x))
            src_y2 = int(min(new_H, H - final_offset_y))
            
            # 确保源图像裁剪区域有效
            src_x1 = min(src_x1, new_W - 1)
            src_y1 = min(src_y1, new_H - 1)
            src_x2 = max(src_x2, src_x1 + 1)
            src_y2 = max(src_y2, src_y1 + 1)
            
            # 处理边缘填充模式
            if pad_mode == "edge" and (src_x1 > 0 or src_y1 > 0 or src_x2 < new_W or src_y2 < new_H):
                # 创建扩展的图像 (增加边缘填充)
                pad_size = max(abs(end_horizontal_shift), abs(end_vertical_shift), 50)  # 安全边界
                padded_scaled = torch.zeros((1, new_H + pad_size*2, new_W + pad_size*2, C), 
                                          dtype=scaled_tensor.dtype, device=scaled_tensor.device)
                
                # 用边缘像素填充
                # 中心区域
                padded_scaled[0, pad_size:pad_size+new_H, pad_size:pad_size+new_W] = scaled_tensor[0]
                
                # 填充边缘
                # 上边缘
                if new_W > 0:
                    padded_scaled[0, :pad_size, pad_size:pad_size+new_W] = scaled_tensor[0, 0:1].expand(pad_size, new_W, C)
                # 下边缘
                if new_W > 0:
                    padded_scaled[0, pad_size+new_H:, pad_size:pad_size+new_W] = scaled_tensor[0, -1:].expand(pad_size, new_W, C)
                # 左边缘
                if new_H > 0:
                    padded_scaled[0, pad_size:pad_size+new_H, :pad_size] = scaled_tensor[0, :, 0:1].expand(new_H, pad_size, C)
                # 右边缘
                if new_H > 0:
                    padded_scaled[0, pad_size:pad_size+new_H, pad_size+new_W:] = scaled_tensor[0, :, -1:].expand(new_H, pad_size, C)
                # 四个角
                if new_W > 0 and new_H > 0:
                    padded_scaled[0, :pad_size, :pad_size] = scaled_tensor[0, 0, 0]  # 左上
                    padded_scaled[0, :pad_size, pad_size+new_W:] = scaled_tensor[0, 0, -1]  # 右上
                    padded_scaled[0, pad_size+new_H:, :pad_size] = scaled_tensor[0, -1, 0]  # 左下
                    padded_scaled[0, pad_size+new_H:, pad_size+new_W:] = scaled_tensor[0, -1, -1]  # 右下
                
                # 调整源图像裁剪区域
                src_x1 += pad_size
                src_y1 += pad_size
                src_x2 += pad_size
                src_y2 += pad_size
                scaled_tensor = padded_scaled
            
            # 粘贴缩放后的图像到画布
            if paste_x2 > paste_x1 and paste_y2 > paste_y1 and src_x2 > src_x1 and src_y2 > src_y1:
                # 从缩放图像中裁剪有效区域
                cropped_scaled = scaled_tensor[0, src_y1:src_y2, src_x1:src_x2, :]
                
                # 粘贴到画布
                canvas[0, paste_y1:paste_y2, paste_x1:paste_x2, :] = cropped_scaled
                
                # 更新蒙版 (0=图像区域)
                mask[0, paste_y1:paste_y2, paste_x1:paste_x2] = 0
            
            output_images.append(canvas)
            output_masks.append(mask)
        
        # 组合结果
        output_images = torch.cat(output_images, dim=0)
        output_masks = torch.cat(output_masks, dim=0)
        
        return (output_images, output_masks)