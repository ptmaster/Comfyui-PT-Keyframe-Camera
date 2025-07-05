from .PT_KeyframeCamera import PT_KeyframeCamera

NODE_CLASS_MAPPINGS = {
    "PT_KeyframeCamera": PT_KeyframeCamera,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PT_KeyframeCamera": "Keyframe Camera (Pan & Zoom)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']