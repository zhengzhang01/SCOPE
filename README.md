<div align="center">

<h1>SCOPE: Scale-Consistent One-Pass Estimation of 3D Geometry</h1>

[**Zheng Zhang**](https://zhengzhang01.github.io/)<sup>1</sup> ·
[**Lihe Yang**](https://liheyoung.github.io/)<sup>1</sup> ·
[**Tianyu Yang**](https://tianyu-yang.com/)<sup>2</sup> ·
[**Chaohui Yu**](https://richardych.github.io/)<sup>2</sup> ·
[**Yixing Lao**](https://yxlao.github.io/)<sup>1</sup> ·
[**Xiaoyang Guo**](https://xy-guo.github.io/)<sup>3</sup> ·
[**Biao Gong**](https://biao-gong.github.io/)<sup>4</sup> ·
[**Fan Wang**](https://scholar.google.com/citations?hl=en&user=WCRGTHsAAAAJ)<sup>2</sup> ·
[**Hengshuang Zhao**](https://hszhao.github.io/)<sup>1</sup>

<sup>1</sup>The University of Hong Kong &emsp;
<sup>2</sup>Alibaba Group &emsp;
<sup>3</sup>Horizon Robotics &emsp;
<sup>4</sup>Ant Group

**SIGGRAPH Conference Papers 2026**

<a href="https://scope3d.github.io/"><img src="https://img.shields.io/badge/Project_Page-SCOPE-2ea44f" alt="Project Page"></a>
<a href="https://doi.org/10.1145/3799902.3811149"><img src="https://img.shields.io/badge/Paper-SIGGRAPH_2026-b31b1b" alt="Paper"></a>
<a href="https://github.com/zhengzhang01/SCOPE"><img src="https://img.shields.io/badge/Code-GitHub-24292f" alt="Code"></a>
<a href="https://huggingface.co/zhengzhang01/SCOPE"><img src="https://img.shields.io/badge/Model_Weights-Hugging_Face-f9d371" alt="Model Weights"></a>
<a href="https://huggingface.co/zhengzhang01/SCOPE/resolve/main/checkpoint.pt"><img src="https://img.shields.io/badge/Checkpoint-checkpoint.pt-1f6feb" alt="Checkpoint"></a>

</div>

SCOPE estimates scale-consistent and temporally stable 3D geometry from monocular videos. This repository contains the model code, training entrypoint, and inference entrypoint used by the released weights.

## Installation

```bash
cd SCOPE
pip install -r requirements.txt
pip install -e .
```

Use a Python environment with PyTorch, CUDA support if you want GPU inference or training, and the dependencies in `requirements.txt`. The repository also contains a bundled `utils3d` copy for environments where the external package is unavailable.

If your environment cannot install `utils3d` from the dependency URL, install the bundled copy:

```bash
pip install -e utils3d
```

## Inference

`scope infer` is the public inference command. It accepts a video file, a directory containing multiple sequence folders, or a single directory of images.

The repository includes a small RGB Baseball example video from the project page at `examples/videos/baseball_rgb.mp4`. Video files are extracted to a runtime image sequence automatically. By default, inference saves both compressed raw predictions and a side-by-side RGB/depth visualization video:

```bash
scope infer \
  --input-dir examples/videos/baseball_rgb.mp4 \
  --output-dir ./scope_predictions \
  --model scope \
  --resolution-level 0 \
  --save-raw-data false
```

```bash
scope infer \
  --input-dir /path/to/sequences \
  --output-dir ./scope_predictions \
  --model scope \
  --resolution-level 3
```

For a single image sequence stored directly as images:

```bash
scope infer \
  --input-dir /path/to/sequence/images \
  --output-dir ./scope_predictions \
  --image-glob "*.png"
```

Long-sequence chunking can be adjusted without editing code:

```bash
SCOPE_INFER_LEN=24 SCOPE_OVERLAP=8 SCOPE_INTERP_LEN=4 scope infer \
  --input-dir /path/to/sequence/images \
  --output-dir ./scope_predictions
```

Outputs are compressed `.npz` files named `<sequence_name>_scope_raw.npz`. Each file contains:

- `depths`: predicted depth maps.
- `masks`: valid prediction masks.
- `intrinsics`: normalized camera intrinsics.
- `original_size`: input image size before resizing.
- `resized_size`: network input size.
- `points`: saved only when `--save-raw-data true`; this can be large.

Visualization videos are saved as `<sequence_name>_scope_vis.mp4` unless `--save-vis-video false` is passed.

The same inference implementation is also available as:

```bash
python -m scope.scripts.infer_video --input-dir /path/to/sequences --output-dir ./scope_predictions
python scope_infer.py --input-dir /path/to/sequences --output-dir ./scope_predictions
```

## Training

The public training command is:

```bash
scope train \
  --config configs/train/scope.json \
  --workspace workspace/scope
```

For distributed training:

```bash
accelerate launch --config_file config/8gpu.yaml \
  scope/scripts/train.py \
  --config configs/train/scope.json \
  --workspace workspace/scope
```

Full training expects dataset split files under `scope/dataset/splits` and local dataset roots matching those split files. The open-source tree includes the loader code and a splits README; restore or regenerate the real split files before launching full dataset training.

Edit `configs/train/scope.json` before full training:

- `data.dataset_roots.GTASFM`: directory containing GTASFM `.hdf5` files.
- `data.dataset_roots.Hypersim`: Hypersim root containing `scenes/metadata_camera_parameters.csv`.
- `data.dataset_roots.IRS`: IRS `extracted` root.
- `data.metadata.Spring.cam_data_base`: optional Spring camera-data root. If omitted, the loader infers it from split paths.
- `data.metadata.LightWheel.info_pickle_paths`: optional LightWheel metadata pickle files.

Training saves final model weights to `workspace/scope/checkpoint/checkpoint.pt`. Periodic training states are saved as `step_XXXXXXXX.pt`.

## Python API

```python
from scope.model import import_model_class
from scope.utils.checkpoints import resolve_checkpoint_path

checkpoint = resolve_checkpoint_path("auto")
ScopeModel = import_model_class("scope")
model = ScopeModel.from_pretrained(checkpoint).cuda().eval()
```

To use a different public or private model repository, pass it explicitly:

```bash
scope infer \
  --input-dir /path/to/sequences \
  --output-dir ./scope_predictions \
  --checkpoint your-org/your-scope-checkpoint
```

For private or gated repositories, set `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` in the environment before running inference or training. Do not commit tokens to this repository.

## Acknowledgements

SCOPE builds on ideas and code structure from [MoGe](https://github.com/microsoft/MoGe), an open-source monocular geometry estimation project from Microsoft. We thank the MoGe authors for making their implementation available.

SCOPE also includes or adapts third-party components under their original licenses, including DINOv2 code from Meta, temporal attention code derived from AnimateDiff/HuggingFace implementations, and a bundled copy of `utils3d`. See `NOTICE` and the license headers in the corresponding source files for details.

## Citation

If you find this project useful, please consider citing:

```bibtex
@inproceedings{zhang2026scope,
  title     = {SCOPE: Scale-Consistent One-Pass Estimation of 3D Geometry},
  author    = {Zhang, Zheng and Yang, Lihe and Yang, Tianyu and Yu, Chaohui and Lao, Yixing and Guo, Xiaoyang and Gong, Biao and Wang, Fan and Zhao, Hengshuang},
  booktitle = {SIGGRAPH Conference Papers},
  year      = {2026}
}
```
