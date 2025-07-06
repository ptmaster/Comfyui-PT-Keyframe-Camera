# Comfyui-PT-Keyframe-Camera
PT Keyframe Camera
# PT Keyframe Camera - 专业级关键帧相机动画节点

> 为ComfyUI设计的强大关键帧相机动画节点，轻松创建好莱坞级的平移和缩放效果
>
> 2025年7月6日 PT Keyframe Camera - 增强版（带旋转功能）
专业级关键帧相机动画节点，新增旋转功能，实现全方位动态控制

✨ 新增旋转功能亮点
🌀 中心点旋转 - 围绕图像中心点进行高质量旋转

↔️ 全方位角度 - 支持-360°到+360°任意角度旋转

⏱️ 平滑旋转过渡 - 关键帧之间自动插值旋转角度

🔄 双向旋转 - 正角度=顺时针旋转，负角度=逆时针旋转

🧩 复合动画 - 旋转与平移、缩放无缝结合，创建复杂运动效果

🎯 高质量插值 - 双线性插值确保旋转平滑无锯齿

🔄 旋转参数说明
参数名	类型	默认值	说明
start_rotation	INT	0	起始帧旋转角度（度），正=顺时针，负=逆时针
end_rotation	INT	0	结束帧旋转角度（度），正=顺时针，负=逆时针

🌟 向右旋转60度,在通义万相WAN中进行骨骼图输出,并通过VACE控制节点最终的效果展示

![QQ_1751776339835](https://github.com/user-attachments/assets/14dbbec0-d723-47d1-8eec-eeafa68f91c4)



https://github.com/user-attachments/assets/1dd60e95-3d04-4167-a264-3a541ad2be16




## ✨ 功能亮点

- 🎬 **专业级关键帧动画** - 在起始帧和结束帧之间平滑过渡相机运动
- 🌐 **全方位位移控制** - 支持左/右/上/下四个方向的精准位移（支持负值）
- 🔍 **动态缩放效果** - 可设置起始帧和结束帧的缩放比例，实现推拉镜头效果
- 🎯 **多种缩放原点** - 支持中心、左上角、右上角、左下角、右下角五种缩放模式
- 🖼️ **智能边缘处理** - 提供颜色填充和边缘扩展两种模式，避免黑边问题
- ⚡ **高效批量处理** - 一键处理整个图像序列，快速生成动画

- 这是一个基座类型的节点,可以尽情释放想象力,输入单张图片复制的帧序列,或者运动视频序列,输出你想要的镜头平移和缩放.
- 完全可以应用到任何视频制作中,无论是直接输出视频,还是生成运动参考视频或者深度和骨骼等控制线条.
- 可以多个节点串联,实现多镜头组合.
- 现在就试试通过数个节点来组合一个肩扛运动相机的镜头效果吧!输出的镜头视频将成为你永久好用的个人数字资产!

## 🎮 使用指南
一个经典的工作流示例:
![workflow](https://github.com/user-attachments/assets/dcd64105-91c9-4b96-996c-d892f371f588)

来自爱屋大佬AIWOOD充满创意的视频科普:
https://www.bilibili.com/video/BV1363dzXEuw/?vd_source=8c0c4ef61dc5c5d2356d3037a952ed3b

![QQ_1751738378004](https://github.com/user-attachments/assets/d6f9fda1-802f-49d2-8175-8b4afadf1c03)

### 节点参数说明

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| **images** | IMAGE | - | 输入图像序列（4D张量：[帧数, 高, 宽, 通道]） |
| **start_frame** | INT | 0 | 动画起始帧号 |
| **end_frame** | INT | 24 | 动画结束帧号 |
| **start_horizontal_shift** | INT | 0 | 起始帧水平位移（正=右，负=左） |
| **start_vertical_shift** | INT | 0 | 起始帧垂直位移（正=下，负=上） |
| **start_zoom** | FLOAT | 1.0 | 起始帧缩放倍数 |
| **end_horizontal_shift** | INT | 100 | 结束帧水平位移（正=右，负=左） |
| **end_vertical_shift** | INT | 0 | 结束帧垂直位移（正=下，负=上） |
| **end_zoom** | FLOAT | 1.0 | 结束帧缩放倍数 |
| **zoom_origin** | LIST | "center" | 缩放中心点（中心/左上角/右上角/左下角/右下角） |
| **pad_mode** | LIST | "color" | 填充模式（颜色/边缘） |
| **bg_color** | STRING | "0, 0, 0" | 背景颜色（RGB值，0-255） |

`

## 🌟 在通义万相WAN中进行骨骼图输出,并通过VACE控制节点最终的效果展示

### 向右平移并缩放效果

https://github.com/user-attachments/assets/6495774f-6067-425b-a65f-da50e6dc7c52

### 向左平移并缩放升高效果

https://github.com/user-attachments/assets/ed12465d-e13a-4e6f-83e8-d80499f69353

### 来自用户 穿靴子的猫 制作的大幅度运镜视频


https://github.com/user-attachments/assets/0e3330b7-47f6-4774-ae8f-725e60483d61



## 🚧 常见问题解答

### Q: 为什么我的图像移动方向不对？
A: 请检查位移参数的符号：
- 水平位移正值 = 向右移动
- 水平位移负值 = 向左移动
- 垂直位移正值 = 向下移动
- 垂直位移负值 = 向上移动

### Q: 如何避免图像边缘出现黑边？
A: 将 `pad_mode` 设置为 "edge"，系统会自动扩展图像边缘

### Q: 支持的最大位移值是多少？
A: 最大位移值由 `MAX_RESOLUTION` 设置（默认为8192），可在代码中调整

**PT Keyframe Camera** © 2025 - 专业级关键帧相机动画解决方案  
由 PTMaster 创建并维护 · 
