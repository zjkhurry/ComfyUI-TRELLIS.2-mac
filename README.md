# TRELLIS.2 ComfyUI Custom Nodes

ComfyUI custom nodes for TRELLIS.2 image-to-3D generation, running natively on Apple Silicon.

## Features

- **Image-to-3D Generation**: Generate 3D meshes from single images using TRELLIS.2
- **PBR Textures**: Automatic baking of base color, metallic, and roughness textures
- **Multiple Pipeline Types**: 512, 1024, 1024_cascade
- **Customizable**: Seed, steps, texture size, background removal options
- **Flexible Model Path**: Specify custom model path via input parameter
- **HuggingFace Integration**: Models downloaded directly from HuggingFace Hub

## Installation

### 1. Copy to ComfyUI
```bash
git clone https://github.com/zjkhurry/ComfyUI-TRELLIS.2-mac.git
mv -R ComfyUI-TRELLIS.2-mac /path/to/ComfyUI/custom_nodes/
```

### 2. Run Setup
```bash
cd /path/to/ComfyUI/custom_nodes/TRELLIS.2-ComfyUI
bash setup.sh
```

This will:
- Clone TRELLIS.2 and deps (if not already present)
- Install Python dependencies
- Build Metal backends (mtlbvh, mtldiffrast, mtlgemm, mtlmesh, o-voxel)
- Apply patches

### 3. Restart ComfyUI
After setup, restart ComfyUI to load the new nodes.

## Usage

### Running ComfyUI with MPS Fallback

For Apple Silicon Macs, run ComfyUI with the MPS fallback environment variable:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 main.py
```

This ensures compatibility with operations not natively supported on MPS.

### TRELLIS.2 Shape Node
- **Input**: IMAGE (single view)
- **Parameters**:
  - `model_path`: Path to TRELLIS.2 model (HuggingFace repo ID or local path)
    - Default: `microsoft/TRELLIS.2-4B` (downloads from HuggingFace)
    - Local: `models/TRELLIS.2-4B`
  - `pipeline_type`: 512, 1024, 1024_cascade
  - `seed`: Random seed (default: 42)
  - `steps`: Diffusion steps (default: 12, range: 1-50)
  - `texture_size`: PBR texture resolution (512, 1024, 2048)
  - `no_texture`: Skip texture baking (export geometry only)
  - `rembg`: Remove background before generation
- **Output**: `glb_path` GLB file will save to `output/` directory

### Workflow
1. Use ComfyUI's built-in image loader
2. Connect to "TRELLIS2Shape" node
3. Set `model_path` parameter (default: microsoft/TRELLIS.2-4B)
4. Configure other parameters
5. Connect output to 3D previewer
6. Run generation
7. Check `output/` directory for generated GLB files
  
The example workflow is `Trellis2Example_workflow.json`

## Requirements

- macOS on Apple Silicon (M1+)
- Python 3.11+
- 24GB+ unified memory
- ComfyUI
- Xcode Metal Toolchain (recommended for Metal acceleration)

## Technical Details

- Uses MPS (Metal Performance Shaders) on Apple Silicon
- Fallback to CPU if MPS not available
- Texture baking via xatlas + KDTree
- PBR texture format: base color, metallic, roughness
- Model path supports HuggingFace repo IDs or local paths

## Output Format

GLB files with PBR textures ready for use in 3D applications.

## Acknowledgments

This project is based on the [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) project by Microsoft and [trellis-mac](https://github.com/shivampkumar/trellis-mac) project. 
Special thanks to the original authors for their excellent work.

Additional macOS-specific optimizations and ComfyUI integration provided by the community.

