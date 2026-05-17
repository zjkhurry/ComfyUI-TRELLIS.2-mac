"""
TRELLIS.2 Shape Generation Node for ComfyUI
"""

import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch
import numpy as np
from PIL import Image
from dataclasses import dataclass
import trimesh
from pathlib import Path

# Add the TRELLIS.2 path
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")

try:
    import flex_gemm

    os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
except (ImportError, RuntimeError):
    os.environ.setdefault("SPARSE_CONV_BACKEND", "none")

sys.path.insert(0, os.path.join(PACKAGE_DIR, "..", "TRELLIS.2"))

try:
    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline
except ImportError as e:
    raise ImportError(f"Failed to import TRELLIS.2 pipeline: {e}")

filename_prefix = "trellis2"


def _next_output_path(prefix: str, extension: str = ".glb") -> Path:
    base_dir = Path(_get_output_path())
    stem = (
        "".join(c if c.isalnum() or c in {"_", "-"} else "_" for c in prefix).strip("_")
        or "hy3d"
    )
    candidate = base_dir / f"{stem}{extension}"
    index = 1
    while candidate.exists():
        candidate = base_dir / f"{stem}_{index}{extension}"
        index += 1
    return candidate


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_output_path() -> str:
    """Resolve output path, supporting relative paths."""
    return os.path.join(PACKAGE_DIR, "..", "..", "..", "output")


class Trellis2ShapeNode:
    CATEGORY = "TRELLIS.2"
    FUNCTION = "generate_shape"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("glb_path",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_path": ("STRING", {"default": "microsoft/TRELLIS.2-4B"}),
                "pipeline_type": (
                    "STRING",
                    {"default": "512", "choices": ["512", "1024", "1024_cascade"]},
                ),
                "seed": ("INT", {"default": 42, "min": 0, "max": 999999999}),
                "steps": ("INT", {"default": 12, "min": 1, "max": 50}),
                "texture_size": (
                    "INT",
                    {"default": 1024, "min": 512, "max": 2048, "step": 512},
                ),
                "no_texture": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "rembg": ("BOOLEAN", {"default": True}),
            },
        }

    def generate_shape(
        self,
        image,
        model_path,
        pipeline_type,
        seed,
        steps,
        texture_size,
        no_texture,
        rembg=True,
    ):
        # Convert tensor to PIL
        if image.ndim == 4:
            image = image[0]

        img_array = image.cpu().numpy()
        if img_array.dtype != np.uint8:
            img_array = (img_array.clip(0.0, 1.0) * 255.0).round().astype("uint8")

        if img_array.shape[-1] == 4:
            pil_image = Image.fromarray(img_array, mode="RGBA")
        else:
            pil_image = Image.fromarray(img_array, mode="RGB")

        # Load pipeline
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_path)
        pipeline.to(torch.device(_pick_device()))

        print(f"Device: {pipeline._device}")
        print(f"Model path: {model_path}")
        print(f"Pipeline type: {pipeline_type}, Steps: {steps}, Seed: {seed}")

        # Generate
        torch.manual_seed(seed)

        # Prepare image
        if rembg and pil_image.mode == "RGB":
            if hasattr(pipeline, "rembg_model"):
                pil_image = pipeline.rembg_model(pil_image)

        out_mesh = pipeline.run(
            image=pil_image,
            seed=seed,
            sparse_structure_sampler_params={"steps": steps},
            shape_slat_sampler_params={"steps": steps},
            tex_slat_sampler_params={"steps": steps},
            pipeline_type=pipeline_type,
        )

        del pipeline

        # Process mesh
        mesh_out = out_mesh[0] if isinstance(out_mesh, list) else out_mesh
        verts = mesh_out.vertices
        faces = mesh_out.faces

        print(f"\nMesh: {verts.shape[0]:,} vertices, {faces.shape[0]:,} triangles")

        if no_texture:

            tm = trimesh.Trimesh(vertices=verts, faces=faces)
            glb_path = _next_output_path(filename_prefix, extension=".glb")
            glb_path.parent.mkdir(parents=True, exist_ok=True)
            tm.export(glb_path)
            mesh = trimesh.load(glb_path, force="mesh")
            mesh.export(glb_path)
            return (str(glb_path),)

        # Apply texture baking

        try:
            import o_voxel.postprocess

            backend = getattr(o_voxel.postprocess, "_BACKEND", None)
            has_dr = getattr(o_voxel.postprocess, "_HAS_DR", False)
            use_metal = backend == "metal" and has_dr
            if use_metal and not getattr(o_voxel.postprocess, "_HAS_FLEX_GEMM", False):
                # o_voxel's _grid_sample_3d fallback returns [B*C, M] but the
                # bake consumes it as [M, C]. Patch it to transpose before the
                # reshape. We avoid installing flex_gemm itself because its
                # import slows the diffusion hot path ~10x on MPS.
                import torch.nn.functional as _F_gs

                def _gs3d_fix(feats, coords, shape, grid, mode="trilinear"):
                    B, C = shape[0], shape[1]
                    D, H, W = shape[2], shape[3], shape[4]
                    device = feats.device
                    dense_vol = torch.zeros(
                        B, C, D, H, W, dtype=feats.dtype, device=device
                    )
                    batch_idx = coords[:, 0].long()
                    cx = coords[:, 1].long()
                    cy = coords[:, 2].long()
                    cz = coords[:, 3].long()
                    dense_vol[batch_idx, :, cx, cy, cz] = feats
                    grid_norm = torch.stack(
                        [
                            grid[..., 2] / (W - 1) * 2 - 1,
                            grid[..., 1] / (H - 1) * 2 - 1,
                            grid[..., 0] / (D - 1) * 2 - 1,
                        ],
                        dim=-1,
                    ).reshape(B, 1, 1, -1, 3)
                    sampled = _F_gs.grid_sample(
                        dense_vol,
                        grid_norm,
                        mode="bilinear",
                        align_corners=True,
                        padding_mode="border",
                    )
                    M = grid.shape[1]
                    return sampled.reshape(B, C, M).permute(0, 2, 1).reshape(B * M, C)

                o_voxel.postprocess._grid_sample_3d = _gs3d_fix
        except (ImportError, AttributeError):
            use_metal = False

        if use_metal:
            try:
                print(
                    f"\nBaking PBR textures via Metal ({texture_size}x{texture_size})..."
                )
                import o_voxel

                # Pre-simplify mesh to avoid mtlbvh crash on large meshes.
                # Target ~200K faces — keeps detail, avoids Metal BVH issues.
                import fast_simplification

                verts_np = mesh_out.vertices.cpu().numpy()
                faces_np = mesh_out.faces.cpu().numpy()
                target_faces = min(200000, len(faces_np))
                if len(faces_np) > target_faces:
                    ratio = 1.0 - (target_faces / len(faces_np))
                    print(
                        f"  Simplifying mesh: {len(faces_np):,} -> ~{target_faces:,} faces"
                    )
                    simp_verts, simp_faces = fast_simplification.simplify(
                        verts_np, faces_np, ratio
                    )
                    simp_verts_t = (
                        torch.from_numpy(simp_verts)
                        .float()
                        .to(mesh_out.vertices.device)
                    )
                    simp_faces_t = torch.from_numpy(simp_faces.astype("int32")).to(
                        mesh_out.faces.device
                    )
                else:
                    simp_verts_t = mesh_out.vertices
                    simp_faces_t = mesh_out.faces

                # Move all mesh tensors to CPU — o_voxel.to_glb mixes device-neutral
                # AABB tensor with mesh tensors; keep everything on CPU to avoid mismatch.
                glb = o_voxel.postprocess.to_glb(
                    vertices=simp_verts_t.cpu(),
                    faces=simp_faces_t.cpu(),
                    attr_volume=mesh_out.attrs.cpu(),
                    coords=mesh_out.coords.cpu(),
                    attr_layout=mesh_out.layout,
                    voxel_size=mesh_out.voxel_size,
                    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                    decimation_target=target_faces,
                    texture_size=texture_size,
                    verbose=True,
                )
                glb_path = _next_output_path(filename_prefix, extension=".glb")
                glb_path.parent.mkdir(parents=True, exist_ok=True)
                glb.export(glb_path)
                mesh = trimesh.load(glb_path, force="mesh")
                mesh.export(glb_path)
                return (str(glb_path),)
            except RuntimeError as e:
                print(f"\n  Metal bake failed: {e}")
                print(f"  Falling back to KDTree texture baker...")
                use_metal = False

        if not use_metal:
            print(
                f"\nBaking PBR textures via KDTree ({texture_size}x{texture_size})...Very slow \n Make sure you have run setup.sh and installed all dependencies properly."
            )

            from ..backends.texture_baker import (
                uv_unwrap,
                bake_texture,
                export_glb_with_texture,
            )

            voxel_coords = mesh_out.coords.cpu().float()
            voxel_attrs = mesh_out.attrs.cpu().float()
            origin = mesh_out.origin.cpu().float()
            vs = mesh_out.voxel_size

            # Simplify for UV unwrap
            import fast_simplification

            target_faces = min(200000, len(faces))
            if len(faces) > target_faces:
                ratio = 1.0 - (target_faces / len(faces))
                bake_verts, bake_faces = fast_simplification.simplify(
                    verts.cpu().numpy(), faces.cpu().numpy(), ratio
                )
            else:
                bake_verts, bake_faces = verts.cpu().numpy(), faces.cpu().numpy()

            # UV unwrap
            new_verts, new_faces, uvs, vmapping = uv_unwrap(
                bake_verts.astype(np.float32), bake_faces.astype(np.uint32)
            )

            # Bake texture
            base_color_img, mr_img, mask = bake_texture(
                new_verts,
                new_faces,
                uvs,
                voxel_coords.numpy(),
                voxel_attrs.numpy(),
                origin.numpy(),
                vs,
                texture_size=texture_size,
            )

            glb_path = _next_output_path(filename_prefix, extension=".glb")
            glb_path.parent.mkdir(parents=True, exist_ok=True)
            export_glb_with_texture(
                new_verts, new_faces, uvs, base_color_img, mr_img, glb_path
            )
            mesh = trimesh.load(glb_path, force="mesh")

            mesh.export(glb_path)
        return (str(glb_path),)
