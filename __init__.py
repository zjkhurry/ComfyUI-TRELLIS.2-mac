"""
ComfyUI Custom Nodes for TRELLIS.2 Image-to-3D Generation
Runs natively on Apple Silicon (MPS/Metal)
"""

from .nodes.trellis2_shape_node import Trellis2ShapeNode

NODE_CLASS_MAPPINGS = {
    "Trellis2Shape": Trellis2ShapeNode,
}

NODE_DISPLAY_NAMES_MAPPINGS = {
    "Trellis2Shape": "TRELLIS.2 Shape",
}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAMES_MAPPINGS"]
