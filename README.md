# FoundDiff - Environment Setup Guide (Working)

## Model Weights

Download the required model weights from: [Google Drive](https://drive.google.com/drive/folders/1B33XyPqC9KkmzmfrCq20-7Xxuf-23PMc?usp=sharing)

You will need:
- `DA-CLIP.pth` - Place in `src/` (or as specified in DA-Diff.py)
- `model-400.pt` - Place in `checkpoints/FoundDiff/sample/`

## Quick Setup (Recommended)

Use the provided `install.yaml` to create the environment:

```bash
conda env create -f install.yaml
conda activate founddiff

# Install pre-built wheels for causal-conv1d and mamba-ssm
pip install causal-conv1d==1.2.2.post1 --no-build-isolation
pip install https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu118torch2.1cxx11abiFalse-cp310-cp310-linux_x86_64.whl

# Verify
python -c "import selective_scan_cuda; print('CUDA kernels loaded successfully!')"
```

## System Requirements

- Linux Platform
- NVIDIA GPU with CUDA 11.8+ support
- Python 3.10 

## Manual Step-by-Step Installation (Alternative)

### 1. Create Conda Environment

If you prefer to use the provided install.yaml, skip to step 4 after creating the environment:

```bash
conda create -n founddiff python=3.10 -y
conda activate founddiff
```

### 2. Install PyTorch 2.1.0 with CUDA 11.8

```bash
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install "numpy<2"
```

### 3. Install CUDA Toolkit 11.8

**Critical**: Use the specific nvidia channel label to get CUDA 11.8 (not newer versions):

```bash
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y
```

Verify installation:
```bash
nvcc --version
# Should show: CUDA compilation tools, release 11.8, V11.8.89
```

### 4. Install Prerequisites

```bash
pip install wheel packaging ninja
pip install causal-conv1d==1.2.2.post1 --no-build-isolation
```

### 5. Install mamba-ssm (Pre-built Wheel)

Download and install the pre-built wheel for CUDA 11.8 + Python 3.10:

```bash
pip install https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu118torch2.1cxx11abiFalse-cp310-cp310-linux_x86_64.whl
```

### 6. Set Environment Variables

Create activation script to ensure torch libraries are found:

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d/
echo 'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export CUDA_HOME="$CONDA_PREFIX"' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export PATH="$CONDA_PREFIX/bin:$PATH"' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

### 7. Verify Installation

```bash
conda activate founddiff
python -c "import selective_scan_cuda; print('CUDA kernels loaded successfully!')"
```

Should output: `CUDA kernels loaded successfully!`

## Additional Dependencies

The project requires several Python packages. Install them:

```bash
pip install kornia einops ipdb pywt accelerate ema-pytorch wandb Augmentor opencv-python lmdb scipy scikit-image matplotlib timm
```

Note: Some packages may require additional system dependencies or may have version conflicts. Install them one by one if needed.

## Running the Project

### Training

```bash
conda activate founddiff
CUDA_VISIBLE_DEVICES=1 python train.py --name FoundDiff --is_train --train_num_steps 400000
```

### Inference

Download the model weights from the Google Drive link above, then run:

```bash
conda activate founddiff
CUDA_VISIBLE_DEVICES=4 python train.py --name FoundDiff --epoch 400 --dataset 2020_seen
```

Ensure:
- `DA-CLIP.pth` is placed in `src/` (or as specified in DA-Diff.py)
- `model-400.pt` is placed in `checkpoints/FoundDiff/sample/`

## Environment Summary

| Package | Version | Source |
|---------|---------|--------|
| Python | 3.10 | conda |
| PyTorch | 2.1.0+cu118 | pytorch index |
| torchvision | 0.16.0+cu118 | pytorch index |
| numpy | 1.26.x | pip |
| CUDA Toolkit | 11.8.0 | nvidia/label/cuda-11.8.0 |
| causal-conv1d | 1.2.2.post1 | pip (pre-built) |
| mamba-ssm | 2.2.2 | GitHub releases |

## Data Format and Structure

### Required Data Format

The code expects data in **NumPy .npy format** (not PNG, JPG, or DICOM). Each file should be a 2D numpy array of shape (512, 512).

```python
# How data is loaded in the code (from pdf_dataset.py)
q_data = np.load(self.q_path_list[index]).astype(np.float32)  # Low dose CT
f_data = np.load(self.f_path_list[index]).astype(np.float32)  # Full dose CT
```

### Required Folder Structure

The code expects data organized in this structure:

#### For Mayo 2020 Dataset (Testing)

```
Mayo2020_ab_2d/test/full_1mm/          # High dose (full dose) - abdominal
Mayo2020_ab_2d/test/quarter_1mm/       # Low dose (1/4 dose) - abdominal

Mayo2020_lung_2d/test/full_1mm/       # High dose - lung
Mayo2020_lung_2d/test/quarter_1mm/     # Low dose - lung

Mayo2020_head_2d_2/test/full_1mm/       # High dose - head
Mayo2020_head_2d_2/test/quarter_1mm/     # Low dose - head
```

#### For Mayo 2016 Dataset

```
Mayo2016_2d/test/full_1mm/           # High dose
Mayo2016_2d/test/quarter_1mm/         # Low dose
```

Or alternatively:
```
Mayo2016_2d/train/full_1mm/          # High dose (training)
Mayo2016_2d/quarter_1mm/            # Low dose
```

### Files to Modify for Custom Data Paths

You need to modify paths in **2 files**:

#### 1. data/pdf_dataset.py (Lines 331-399)

Key paths to change:
```python
# Line 331-340: Abdominal data
ab_ndct = sorted_list('/mnt/miah203/zhchen/Mayo2020_ab_2d/'+ phase+'/full_1mm/*')[:num]
ab_dose_1_4_list = sorted_list('/mnt/miah203/zhchen/Mayo2020_ab_2d/'+phase+'/quarter_1mm/*')[start:num:stride]

# Line 354-363: Lung data
lung_ndct = sorted_list('/mnt/miah203/zhchen/Mayo2020_lung_2d/'+ phase+'/full_1mm/*')[:num]
lung_dose_1_4_list = sorted_list('/mnt/miah203/zhchen/Mayo2020_lung_2d/'+phase+'/quarter_1mm/*')[start:num:stride]

# Line 382-391: Head data
head_ndct = sorted_list('/mnt/miah203/zhchen/Mayo2020_head_2d_2/'+ phase+'/full_1mm/*')[:num]
head_dose_1_4_list = sorted_list('/mnt/miah203/zhchen/Mayo2020_head_2d_2/'+phase+'/quarter_1mm/*')[start:num:stride]

# Line 398-399: Mayo2016 data
self.mayo16_ldct = sorted_list('/mnt/miah203/zhchen/Mayo2016_2d/quarter_1mm/*')
self.mayo16_ndct_path_list = sorted_list('/mnt/miah203/zhchen/Mayo2016_2d/train/full_1mm/*')
```

#### 2. data/mayo16_dataset.py (Lines 42-49)

```python
# Line 42-43: Test data paths
self.f_path_list = sorted_list('/mnt/miah203/zhchen/CQ500_2d/test/full_1mm/*')
self.q_path_list = sorted_list('/mnt/miah203/zhchen/CQ500_2d/test/sim-0.25/*')

# Line 46-48: Training data paths
self.q_path_list = sorted_list('/mnt/miah203/zhchen/Mayo2016_2d/test/quarter_1mm/*')
self.f_path_list = sorted_list('/mnt/miah203/zhchen/Mayo2016_2d/test/full_1mm/*')
```

### Example: Setting Custom Data Path

To use your own data at `/home/user/my_data/`, change the paths in pdf_dataset.py:

```python
# Before:
ab_ndct = sorted_list('/mnt/miah203/zhchen/Mayo2020_ab_2d/'+ phase+'/full_1mm/*')[:num]

# After:
ab_ndct = sorted_list('/home/user/my_data/Mayo2020_ab_2d/'+ phase+'/full_1mm/*')[:num]
```

### Data Filename Matching

The code matches low-dose and high-dose files by index:
- Low dose file: `ab-001.npy` → loads from `quarter_1mm/`
- High dose file: `001.npy` → loads from `full_1mm/` using index from filename

Ensure your low-dose and high-dose files have matching indices in their filenames.

### Converting DICOM to .npy

If your data is in DICOM format (.IMA, .dcm), you need to convert it:

```python
import numpy as np
import pydicom
from PIL import Image
import os

def dicom_to_npy(dicom_path, output_path):
    """Convert DICOM to .npy format"""
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array
    
    # Resize to 512x512 if needed
    if img.shape != (512, 512):
        img = Image.fromarray(img).resize((512, 512))
        img = np.array(img)
    
    # Normalize to HU values if needed
    # img = img * ds.RescaleSlope + ds.RescaleIntercept
    
    np.save(output_path, img.astype(np.float32))

# Example usage:
for f in os.listdir('/path/to/dicom/files'):
    if f.endswith('.dcm') or f.endswith('.IMA'):
        dicom_to_npy(f, f.replace('.dcm', '.npy'))
```

## Troubleshooting

### "libc10.so: cannot open shared object file"
- Ensure the activation script in Step 6 is created correctly
- Or manually run: `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"`

### "nvcc shows CUDA 13.2"
- You installed cuda-toolkit without specifying the channel label
- Recreate environment and use: `conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y`

### mamba-ssm compilation fails
- Always use pre-built wheels from GitHub releases
- Building from source requires specific toolchain and is not recommended

## Credits

- FoundDiff: "Foundational Diffusion Model for Generalizable Low-Dose CT Denoising"
- Mamba SSM: https://github.com/state-spaces/mamba