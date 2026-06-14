# SCOPE

SCOPE estimates scale-consistent monocular video geometry from image sequences. This open-source package contains the SCOPE model code, the public training entrypoint, and the public inference entrypoint used by the released checkpoint.

- Project page: [https://scope3d.github.io/](https://scope3d.github.io/)
- Paper: [https://scope3d.github.io/](https://scope3d.github.io/)

## Checkpoint

By default, SCOPE first looks for a local checkpoint at:

```bash
../checkpoint/checkpoint.pt
```

From this directory, that resolves to the sibling `checkpoint` folder in the release bundle. If the file is not present, SCOPE automatically downloads `checkpoint.pt` from the Hugging Face model repository `zhengzhang01/SCOPE` and reuses the local Hugging Face cache on later runs.

You can override the checkpoint source with `--checkpoint /path/to/checkpoint.pt` or `--checkpoint <huggingface_repo_id>`.

## Installation

```bash
cd scope
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

Training saves the final model weights to `workspace/scope/checkpoint/checkpoint.pt`. Periodic checkpoints are saved as `step_XXXXXXXX.pt`.

### Smoke Test

Use the smoke test to validate the training code path without external datasets. It builds a synthetic 24-frame batch, loads the checkpoint, runs one forward/backward/update step, and writes logs/checkpoint metadata to the workspace.

```bash
scope train \
  --smoke-test \
  --workspace /tmp/scope_smoke_train \
  --epochs 1 \
  --batch_size_forward 1 \
  --gradient_accumulation_steps 1 \
  --enable_mlflow False \
  --enable_ema False \
  --enable_gradient_checkpointing False \
  --enable_mixed_precision True \
  --save_every 1000 \
  --log_every 1 \
  --vis_every 0
```

## Python API

```python
from pathlib import Path
from scope.model import import_model_class

checkpoint = Path("../checkpoint/checkpoint.pt")
ScopeModel = import_model_class("scope")
model = ScopeModel.from_pretrained(checkpoint).cuda().eval()
```

To use the hosted checkpoint in Python, resolve it first:

```python
from scope.utils.checkpoints import resolve_checkpoint_path

checkpoint = resolve_checkpoint_path("auto")
```

## Model Weights

The released checkpoint is hosted in the Hugging Face model repository `zhengzhang01/SCOPE` as `checkpoint.pt`.

The default inference and training commands use this checkpoint automatically when no local checkpoint is found. To use a different public or private model repository, pass it explicitly:

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
