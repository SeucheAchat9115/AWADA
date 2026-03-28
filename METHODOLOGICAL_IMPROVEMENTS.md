# Methodological Improvements for AWADA

This document captures suggested improvements identified after a systematic scan of the
repository.  Each item includes the affected file(s), the current behaviour, the
proposed change, and the rationale.  Improvements are grouped by theme and ordered
roughly from most impactful to least.

---

## 1. Training Stability & Reproducibility

### 1.1 Add a global random-seed utility and apply it in every training script

**Affected files:** `tools/train_*.py`, `tools/generate_attention_maps.py`

**Current behaviour:** No seed is set anywhere.  Results differ between runs, making
ablations and comparisons unreliable.

**Proposed change:** Add a `set_seed(seed: int)` helper in `awada/utils/train_utils.py`
and call it at the top of every `main()` function:

```python
# awada/utils/train_utils.py
import random, numpy as np, torch

def set_seed(seed: int = 42) -> None:
    """Set Python / NumPy / PyTorch seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

Expose it as a `--seed` CLI argument in each training script (default `42`).

**Rationale:** Reproducibility is a cornerstone of empirical ML research.  Seeding
costs nothing at runtime and makes every reported result reproducible with a single flag.

---

### 1.2 Detect and warn on NaN / Inf losses during training

**Affected files:** `tools/train_awada.py`, `tools/train_cyclegan.py`,
`tools/train_cycada.py`

**Current behaviour:** If a loss becomes `NaN` or `Inf` (e.g. due to a degenerate
attention mask of all zeros), `backward()` silently poisons all gradients and training
continues with corrupted weights.

**Proposed change:** Add a lightweight check after each loss computation:

```python
if not torch.isfinite(g_losses["total_G"]):
    raise RuntimeError(
        f"Non-finite generator loss at epoch {epoch+1}, iter {iteration+1}: "
        f"{g_losses['total_G'].item()}"
    )
```

Apply the same check to discriminator losses.

**Rationale:** Failing fast on degenerate losses saves hours of wasted compute and
surfaces subtle bugs (e.g. all-zero attention masks producing undefined masked MSE).

---

### 1.3 Save optimiser state in CycleGAN / CyCada checkpoints for resumable training

**Affected files:** `tools/train_cyclegan.py`, `tools/train_cycada.py`,
`tools/train_awada.py`

**Current behaviour:** Only model weights are saved.  Resuming from a checkpoint
restarts the learning-rate schedule from epoch 0 and loses the Adam momentum state,
producing a different loss trajectory than an uninterrupted run.

**Proposed change:** Include optimiser and scheduler states in every checkpoint:

```python
ckpt = {
    "epoch": epoch + 1,
    "G_AB": model.G_AB.state_dict(),
    "G_BA": model.G_BA.state_dict(),
    "D_A":  model.D_A.state_dict(),
    "D_B":  model.D_B.state_dict(),
    "opt_G": opt_G.state_dict(),
    "opt_D": opt_D.state_dict(),
    "sched_G": sched_G.state_dict(),
    "sched_D": sched_D.state_dict(),
}
```

Add a `--resume` CLI flag that loads the checkpoint and calls
`opt.load_state_dict(...)` / `sched.load_state_dict(...)` before the training loop.

**Rationale:** Without this, a job preempted at epoch 150 of 200 cannot be resumed
correctly—the LR schedule will produce a wrong learning rate for the remaining epochs.

---

### 1.4 Replace `print()` calls with Python's `logging` module

**Affected files:** `tools/train_*.py`, `awada/utils/attention.py`

**Current behaviour:** All progress and diagnostic output uses bare `print()`.  This
makes it impossible to suppress library-level messages, redirect output to a file, or
set verbosity levels without modifying source code.

**Proposed change:**

```python
# awada/utils/attention.py  (and all training scripts)
import logging
logger = logging.getLogger(__name__)

# Replace:
print(f"Saved {len(dataloader.dataset)} attention maps to {output_dir}")
# With:
logger.info("Saved %d attention maps to %s", len(dataloader.dataset), output_dir)
```

Add `logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")`
at the start of each `main()`.

**Rationale:** The `logging` module is the standard Python mechanism for structured,
controllable output.  It enables log redirection (`--log-file`), verbosity control
(`--quiet`), and downstream tooling (log aggregators, TensorBoard writers) without
touching call sites.

---

## 2. Code Quality & Type Safety

### 2.1 Complete the return-type annotations on all public functions

**Affected files:** `awada/models/*.py`, `awada/utils/*.py`, `awada/datasets/*.py`

**Current behaviour:** Many functions lack return-type annotations, reducing the
value of the MyPy pass and making the API harder to understand at a glance.

**Examples of missing annotations:**

| Function | File | Return Type |
|---|---|---|
| `CycleGAN.forward` | `cyclegan.py` | `-> None` |
| `CycleGAN.compute_generator_loss` | `cyclegan.py` | `-> dict[str, torch.Tensor]` |
| `CycleGAN.compute_discriminator_loss` | `cyclegan.py` | `-> dict[str, torch.Tensor]` |
| `CycleGAN.set_input` | `cyclegan.py` | `-> None` |
| `ImageBuffer.push_and_pop` | `cyclegan.py` | `-> torch.Tensor` |
| `generate_attention_maps` | `attention.py` | `-> None` |
| `ResNetGenerator.forward` | `generator.py` | `-> torch.Tensor` |
| `AttentionPairedDataset.__getitem__` | `attention_dataset.py` | `-> tuple[...]` |

**Proposed change:** Annotate all public functions consistently.  `mypy` is already
configured—stricter annotations will surface real bugs for free.

**Rationale:** Complete type annotations are a low-effort, high-value improvement that
enables static analysis to catch real bugs (e.g. passing a CPU tensor where a GPU
tensor is expected) and improves IDE auto-completion for contributors.

---

### 2.2 Tighten the MyPy configuration

**Affected file:** `pyproject.toml`

**Current behaviour:**

```toml
[tool.mypy]
python_version = "3.11"
ignore_missing_imports = true
```

`ignore_missing_imports = true` suppresses type errors from third-party packages that
do not ship stubs, but it also suppresses legitimate errors (e.g. calling a
non-existent method on a `torch.Tensor`).

**Proposed change:** Keep `ignore_missing_imports = true` for external libraries but
enable stricter checks for the project's own code:

```toml
[tool.mypy]
python_version = "3.11"
ignore_missing_imports = true
# Enable stricter checks for first-party code
check_untyped_defs = true
warn_return_any = true
warn_unused_ignores = true
```

**Rationale:** These three flags catch a large class of real bugs (e.g. functions that
return `Any` implicitly, ignored errors that are no longer needed) without requiring
stub packages for PyTorch.

---

### 2.3 Extract magic numbers into named constants

**Affected files:** `awada/models/cyclegan.py`, `tools/train_*.py`

**Current behaviour:** Several meaningful constants appear as inline literals:

| Value | Location | Meaning |
|---|---|---|
| `50` | `cyclegan.py:15` | Image replay buffer size |
| `0.5` | `cyclegan.py:27` | Probability to return buffered image |
| `100` | `train_awada.py:130` | Loss logging interval (iterations) |
| `0.5` | `cyclegan.py:103` | Discriminator loss averaging factor |

**Proposed change:** Define module-level constants and reference them by name:

```python
# cyclegan.py
_BUFFER_SIZE: int = 50          # Zhu et al. (2017) default
_BUFFER_REPLAY_PROB: float = 0.5

# train_awada.py  (or train_utils.py)
_LOG_INTERVAL: int = 100        # iterations between loss printouts
```

**Rationale:** Named constants make intent explicit, ease experimentation (change one
place instead of grepping for the literal), and prevent "why is this 50?" questions
during code review.

---

### 2.4 Guard the `device` default against missing CUDA in all models

**Affected files:** `awada/models/cyclegan.py`, `awada/models/cycada.py`,
`awada/models/awada.py`, `awada/models/semantic_loss.py`

**Current behaviour:** Every model constructor defaults to `device="cuda"`.  When
imported on a CPU-only machine (e.g. CI, macOS laptops), instantiating a model without
an explicit `device` argument silently places tensors on CUDA and raises a
`RuntimeError` later.

**Proposed change:** Use a module-level default that respects the environment:

```python
import torch

_DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class CycleGAN(nn.Module):
    def __init__(self, device: str = _DEFAULT_DEVICE) -> None:
        ...
```

**Rationale:** The training scripts already do this correctly (e.g. `train_detector.py`
line 81).  Applying the same pattern to the model classes makes them safe to import and
unit-test on any machine, which is also why the CI runs on CPU.

---

## 3. Error Handling & Robustness

### 3.1 Validate required directories early in dataset constructors

**Affected files:** `awada/datasets/*.py`, `awada/datasets/attention_dataset.py`

**Current behaviour:** If `source_root` or `attention_root` do not exist,
`os.listdir()` raises a `FileNotFoundError` with a terse OS-level message.  The user
sees no indication of which argument is wrong or what the expected structure is.

**Proposed change:** Add explicit checks at the top of `__init__`:

```python
if not os.path.isdir(source_root):
    raise FileNotFoundError(
        f"Source image directory not found: {source_root!r}. "
        "Run Stage A (train_detector.py) before AWADA training."
    )
if not os.path.isdir(attention_root):
    raise FileNotFoundError(
        f"Attention map directory not found: {attention_root!r}. "
        "Run generate_attention_maps.py on the source domain first."
    )
```

**Rationale:** Fail-fast with actionable error messages avoids multi-minute confusion
when a user inadvertently swaps argument order or forgets to run a prerequisite stage.

---

### 3.2 Handle `DeepLabV3` weight download failures gracefully

**Affected file:** `awada/models/semantic_loss.py`

**Current behaviour:** If the pretrained weights cannot be downloaded (no internet
access, firewall, quota exceeded), `deeplabv3_resnet50(weights=...)` raises a generic
`urllib.error.URLError` or `RuntimeError` with no guidance on how to resolve it.

**Proposed change:**

```python
try:
    net = deeplabv3_resnet50(weights=weights)
except Exception as exc:
    raise RuntimeError(
        "Failed to download DeepLabV3-ResNet50 weights. "
        "Pre-download them with: "
        "`python -c \"from torchvision.models.segmentation import "
        "deeplabv3_resnet50, DeepLabV3_ResNet50_Weights; "
        "deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)\"`"
        " or set TORCH_HOME to a directory containing the cached weights."
    ) from exc
```

**Rationale:** This is a common failure mode in cluster environments with restricted
internet access.  A single actionable message saves significant debugging time.

---

### 3.3 Warn when an attention map is missing for a source image

**Affected file:** `awada/datasets/attention_dataset.py`

**Current behaviour:** `_load_attention()` silently returns `None` when no `.npy` file
is found, falling back to an all-ones mask.  This means a partially-generated attention
directory is indistinguishable from a fully-generated one.

**Proposed change:**

```python
import warnings

def _load_attention(self, img_path: str, attention_root: str) -> np.ndarray | None:
    filename_stem = os.path.splitext(os.path.basename(img_path))[0]
    npy_path = os.path.join(attention_root, filename_stem + ".npy")
    if os.path.exists(npy_path):
        return np.load(npy_path).astype(np.float32)
    warnings.warn(
        f"Attention map not found for {os.path.basename(img_path)!r}; "
        "falling back to uniform (all-ones) mask.  "
        "Run generate_attention_maps.py to create missing maps.",
        UserWarning,
        stacklevel=3,
    )
    return None
```

**Rationale:** Silent degradation makes bugs very hard to find.  A `UserWarning` is
visible in the training log without stopping the job, alerting the user to a
misconfigured dataset.

---

### 3.4 Add a YAML config schema validation step

**Affected file:** `awada/utils/train_utils.py`

**Current behaviour:** `load_config()` returns whatever is in the YAML file.  Typos
in key names (e.g. `lamba_cyc`—note the missing `d`—instead of `lambda_cyc`) are silently ignored and the
code falls back to hardcoded defaults, producing unexpected results with no error.

**Proposed change:** Accept an optional set of required keys and validate them:

```python
_REQUIRED_KEYS = {"epochs", "lr", "betas", "lambda_gan", "lambda_cyc", "batch_size"}

def load_config(path: str, required_keys: set[str] | None = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if required_keys:
        missing = required_keys - cfg.keys()
        if missing:
            raise ValueError(
                f"Config file {path!r} is missing required keys: {sorted(missing)}"
            )
    unknown = cfg.keys() - _ALL_KNOWN_KEYS  # define once in train_utils.py
    if unknown:
        import warnings
        warnings.warn(f"Unknown config keys ignored: {sorted(unknown)}", UserWarning)
    return cfg
```

**Rationale:** Config validation is the cheapest way to catch typos before they waste
a 200-epoch training run.

---

## 4. Performance & Memory Efficiency

### 4.1 Store attention maps as `uint8` instead of `float32`

**Affected files:** `awada/utils/attention.py`,
`awada/datasets/attention_dataset.py`

**Current behaviour:** Attention maps are binary (0.0 or 1.0) but saved and loaded as
`float32`, using 4× the necessary storage and memory bandwidth.

**Proposed change:**

```python
# attention.py – when saving:
np.save(out_path, attention_map.astype(np.uint8))

# attention_dataset.py – when loading:
return np.load(npy_path).astype(np.float32)   # cast once, at load time
```

**Rationale:** For a 1080×1920 image, `float32` needs 8 MB per map; `uint8` needs
2 MB.  Across 15 000 sim10k images this saves ~90 GB of disk space and speeds up
data loading.

---

### 4.2 Use `bilinear` interpolation (with `align_corners=False`) for attention masks

**Affected file:** `awada/models/awada.py`, line 31

**Current behaviour:**

```python
mask_resized = F.interpolate(mask, size=pred.shape[2:], mode="nearest")
```

`nearest` interpolation on a binary mask can produce block-shaped foreground regions
with hard step edges at the discriminator output resolution, potentially causing
gradient discontinuities.

**Proposed change:**

```python
mask_resized = F.interpolate(
    mask, size=pred.shape[2:], mode="bilinear", align_corners=False
)
```

This produces a soft spatial weighting that transitions smoothly between foreground and
background regions.

**Rationale:** Smooth spatial gradients improve GAN training stability.  The mask is
already binary, so a soft interpolation only affects the spatial boundary regions—
exactly where smooth weighting helps most.

---

### 4.3 Pin DataLoader workers on GPU machines

**Affected files:** `tools/train_cyclegan.py`, `tools/train_cycada.py`

**Current behaviour:** `DataLoader` in `train_cyclegan.py` and `train_cycada.py` does
not set `pin_memory`.  `train_awada.py` already sets `pin_memory=True`.

**Proposed change:** Standardise across all training scripts:

```python
DataLoader(
    dataset, batch_size=batch_size, shuffle=True,
    num_workers=4, pin_memory=True, drop_last=True
)
```

**Rationale:** `pin_memory=True` enables faster host-to-device transfers by using
page-locked memory.  It is already applied in `train_awada.py`; applying it everywhere
is a consistency fix.

---

### 4.4 Cache the target file list in `AttentionPairedDataset` by index, not by `random.choice`

**Affected file:** `awada/datasets/attention_dataset.py`

**Current behaviour:** `__getitem__` calls `random.choice(self.target_files)` on every
access, so the target domain is sampled uniformly but independently of the dataset index.
This means `__len__` returns `len(source_files)`, but each epoch visits every target
image only in expectation, not exactly once.

**Proposed change:**

```python
def __getitem__(self, idx: int):
    src_path = self.source_files[idx]
    # Cycle the shorter list so every target image appears roughly once per epoch
    tgt_idx = idx % len(self.target_files)
    tgt_path = self.target_files[tgt_idx]
    ...
```

This is identical to the approach already used in `UnpairedImageDataset`
(`unpaired_dataset.py`).

**Rationale:** Deterministic cycling ensures that all target images are seen each epoch
(given a full pass over the dataset with `shuffle=True`), improving coverage of the
target distribution.

---

## 5. Configuration & Hyperparameter Management

### 5.1 Add a `--save_every` flag to control checkpoint frequency

**Affected files:** `tools/train_awada.py`, `tools/train_cyclegan.py`,
`tools/train_cycada.py`

**Current behaviour:** A checkpoint is saved after every epoch.  For 200-epoch runs on
large datasets this can produce tens of gigabytes of `.pth` files.

**Proposed change:** Add `--save_every N` (default `10`) to each training script and
save only on those epochs plus the final one:

```python
if (epoch + 1) % args.save_every == 0 or epoch == epochs - 1:
    torch.save(ckpt, ckpt_path)
```

**Rationale:** Disk space is finite.  Even with a default of every 10 epochs the user
retains full training coverage while reducing checkpoint footprint by 10×.

---

### 5.2 Add `*.pth` and `runs/` to `.gitignore`

**Affected file:** `.gitignore`

**Current behaviour:** The `.gitignore` does not exclude PyTorch checkpoint files
(`*.pth`) or common experiment output directories.  A user running any training script
can accidentally commit hundreds of megabytes.

**Proposed change:**

```gitignore
# Model checkpoints
*.pth
*.pt
checkpoints/
runs/
outputs/

# Attention maps
*.npy
```

**Rationale:** Checkpoint files are large binary artefacts that do not belong in
version control.  Adding them to `.gitignore` prevents accidental commits and keeps the
repository lightweight.

---

### 5.3 Validate that `lambda_sem > 0` requires attention maps to exist at startup

**Affected file:** `tools/train_awada.py`

**Current behaviour:** A user can run `train_awada.py --lambda_sem 1.0` without
providing `--target_attention_dir`.  The model will instantiate `DeepLabV3` but the
target attention masks will silently default to all-ones, defeating the purpose.

**Proposed change:** Add an explicit check after argument parsing:

```python
if lambda_sem > 0 and args.target_attention_dir is None:
    import warnings
    warnings.warn(
        "--lambda_sem > 0 but --target_attention_dir was not provided. "
        "Target attention maps will default to all-ones. "
        "Pass --target_attention_dir to use semantic-aware target masks.",
        UserWarning,
    )
```

**Rationale:** This is a common misconfiguration that silently produces suboptimal
results with no error message.

---

## 6. Testing & Evaluation

### 6.1 Add a smoke-test for the full training loop (short run)

**Affected file:** `tests/` (new file `tests/test_training_smoke.py`)

**Current behaviour:** Tests cover individual components (models, datasets, metrics)
but there is no end-to-end smoke test that exercises a full training iteration—the most
common path through the code.

**Proposed change:** Add a single test that runs two training iterations with
synthetic data and checks that losses decrease or at least remain finite:

```python
def test_awada_training_step():
    model = AWADA(device="cpu")
    opt_G = torch.optim.Adam(...)
    opt_D = torch.optim.Adam(...)
    real_A = torch.randn(1, 3, 32, 32)
    real_B = torch.randn(1, 3, 32, 32)
    att = torch.ones(1, 1, 32, 32)
    model.set_input(real_A, real_B, att, att)
    model.forward()
    g_losses = model.compute_generator_loss()
    assert torch.isfinite(g_losses["total_G"])
    g_losses["total_G"].backward()
    opt_G.step()
```

**Rationale:** This kind of smoke test prevents regressions in the training pipeline
that unit tests on individual components would not catch (e.g. a shape mismatch only
triggered by combining the generator and discriminator).

---

### 6.2 Add per-class mAP reporting to `evaluate_detector.py`

**Affected file:** `tools/evaluate_detector.py`, `awada/utils/metrics.py`

**Current behaviour:** Evaluation reports only mean mAP@0.5 and mAP@0.5:0.95.  For
multi-class datasets (Cityscapes: 8 classes, BDD100K: 7 classes) it is impossible to
identify which classes benefit or suffer from domain adaptation.

**Proposed change:** Extend `compute_map_range` to also return per-class AP:

```python
# After coco_eval.summarize():
per_class_ap = {}
for cat_id in range(1, num_classes + 1):
    cat_idx = [i for i, e in enumerate(coco_eval.evalImgs) if e and e["category_id"] == cat_id]
    # ... extract per-category precision
```

Return it as `results["per_class_AP"]` and print it in the evaluation script.

**Rationale:** Domain adaptation often degrades performance on rare classes even while
improving the aggregate mAP.  Per-class reporting is essential for understanding model
behaviour and is standard practice in detection benchmarks.

---

### 6.3 Use `pytest.mark.parametrize` to consolidate repeated test cases

**Affected files:** `tests/test_cyclegan.py`, `tests/test_cycada.py`,
`tests/test_awada.py`

**Current behaviour:** Many tests follow the same pattern but are written as separate
functions with copy-pasted setup code.

**Proposed change:** Consolidate with `@pytest.mark.parametrize`:

```python
@pytest.mark.parametrize("lambda_cyc,lambda_gan", [(10.0, 1.0), (5.0, 2.0), (0.0, 1.0)])
def test_generator_loss_finite(lambda_cyc, lambda_gan):
    model = CycleGAN(device="cpu")
    ...
    losses = model.compute_generator_loss(lambda_cyc=lambda_cyc, lambda_gan=lambda_gan)
    assert torch.isfinite(losses["total_G"])
```

**Rationale:** Parametrized tests are easier to extend, more readable, and provide
better coverage without duplicating setup logic.

---

## 7. Architecture & Research Extensions

### 7.1 Support mixed-precision training (AMP)

**Affected files:** `tools/train_*.py`

**Current behaviour:** All training runs in FP32.  On modern NVIDIA GPUs (Ampere and
later), FP32 uses 2× the memory and time of FP16 with negligible accuracy difference
for GAN training.

**Proposed change:** Add `--amp` flag and wrap the forward/backward passes with
`torch.cuda.amp.autocast` and `GradScaler`:

```python
scaler_G = torch.cuda.amp.GradScaler(enabled=args.amp)
scaler_D = torch.cuda.amp.GradScaler(enabled=args.amp)

with torch.cuda.amp.autocast(enabled=args.amp):
    model.forward()
    g_losses = model.compute_generator_loss(...)
scaler_G.scale(g_losses["total_G"]).backward()
scaler_G.step(opt_G)
scaler_G.update()
```

**Rationale:** AMP typically provides a 1.5–2× training speedup and allows doubling the
batch size on the same GPU, both of which matter for a 200-epoch GAN run.

---

### 7.2 Replace the monkey-patched RPN hook with a `register_forward_hook`

**Affected file:** `awada/utils/attention.py`

**Current behaviour:** `generate_attention_maps` replaces
`detector.rpn.filter_proposals` at runtime.  If the torchvision RPN implementation
changes its internal method name or signature, this will silently fail or produce
incorrect maps.

**Proposed change:** Use PyTorch's stable hook API:

```python
captured = {}

def _hook(module, input, output):
    # output is (boxes, scores) from filter_proposals
    captured["boxes"]  = output[0]
    captured["scores"] = output[1]

handle = detector.rpn.register_forward_hook(_hook)
try:
    with torch.no_grad():
        for batch in dataloader:
            ...
            _ = detector(images)
            boxes_list  = captured.get("boxes", [])
            scores_list = captured.get("scores", [])
            ...
finally:
    handle.remove()
```

**Rationale:** `register_forward_hook` is a stable, public PyTorch API that will
continue to work across torchvision versions.  Monkey-patching a private method is a
maintenance liability.

---

### 7.3 Consider soft (continuous) attention maps instead of hard binary masks

**Affected files:** `awada/utils/attention.py`, `awada/datasets/attention_dataset.py`,
`awada/models/awada.py`

**Current behaviour:** Attention maps are thresholded to binary (0 or 1) at
`score_threshold=0.5`.  Foreground regions receive a weight of 1.0 and background
regions a weight of 0.0, with no gradation.

**Proposed change:** Store the raw normalised objectness scores as the attention map,
optionally scaled to [0, 1]:

```python
# In generate_attention_maps – accumulate max score per pixel
for box, score in zip(boxes_np, scores_np):
    x1, y1, x2, y2 = ...
    attention_map[y1:y2, x1:x2] = np.maximum(
        attention_map[y1:y2, x1:x2], float(score)
    )
```

The threshold (`score_threshold`) then becomes a clipping floor rather than a hard
binarisation:

```python
attention_map = np.clip(attention_map, score_threshold, 1.0)
attention_map = (attention_map - score_threshold) / (1.0 - score_threshold)
```

**Rationale:** Soft attention weights provide a more informative signal to the
discriminator—highly confident foreground regions receive higher weight than marginal
proposals—and smooth out the sharp foreground/background boundary, potentially
improving convergence.

---

### 7.4 Log training metrics to TensorBoard or W&B for experiment tracking

**Affected files:** `tools/train_*.py`

**Current behaviour:** Loss values are printed to stdout at fixed intervals and not
persisted anywhere.  Comparing multiple runs (e.g. CycleGAN vs CyCada vs AWADA)
requires manually parsing log files.

**Proposed change:** Add optional TensorBoard logging behind a `--log_dir` flag:

```python
# torch.utils.tensorboard ships with PyTorch (since 1.1),
# but TensorBoard itself must be installed separately: pip install tensorboard
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir=args.log_dir) if args.log_dir else None

# Inside training loop:
global_step = epoch * len(dataloader) + iteration
if writer:
    writer.add_scalar("loss/G_total", g_losses["total_G"].item(), global_step)
    writer.add_scalar("loss/D_total", d_losses["total_D"].item(), global_step)
    writer.add_scalar("loss/cycle",
        (g_losses["cycle_A"] + g_losses["cycle_B"]).item(), global_step)
```

**Rationale:** Experiment tracking is essential for ablation studies.  `torch.utils.tensorboard`
is bundled with PyTorch; only `tensorboard` itself needs to be added to the dev
dependencies (`pip install tensorboard`).  W&B is an alternative if richer experiment
management is needed.

---

## Summary Table

| # | Category | Impact | Effort |
|---|---|---|---|
| 1.1 | Global random seed | High | Low |
| 1.2 | NaN / Inf loss guard | High | Low |
| 1.3 | Resumable checkpoints | High | Medium |
| 1.4 | Replace `print` with `logging` | Medium | Low |
| 2.1 | Complete type annotations | Medium | Low |
| 2.2 | Tighten MyPy config | Medium | Low |
| 2.3 | Named constants | Low | Low |
| 2.4 | CUDA-safe device default | High | Low |
| 3.1 | Early directory validation | High | Low |
| 3.2 | DeepLabV3 download error handling | Medium | Low |
| 3.3 | Warn on missing attention maps | Medium | Low |
| 3.4 | YAML schema validation | Medium | Low |
| 4.1 | `uint8` attention storage | Medium | Low |
| 4.2 | Bilinear mask interpolation | Medium | Low |
| 4.3 | Consistent `pin_memory` | Low | Low |
| 4.4 | Deterministic target cycling | Low | Low |
| 5.1 | `--save_every` checkpoint flag | Medium | Low |
| 5.2 | `.gitignore` for `.pth` and `.npy` | Medium | Low |
| 5.3 | Validate `lambda_sem` + attention dir | Medium | Low |
| 6.1 | End-to-end smoke test | High | Medium |
| 6.2 | Per-class mAP reporting | High | Medium |
| 6.3 | `pytest.mark.parametrize` | Low | Low |
| 7.1 | Mixed-precision training (AMP) | High | Medium |
| 7.2 | Hook API instead of monkey-patch | Medium | Low |
| 7.3 | Soft attention maps | Medium | Medium |
| 7.4 | TensorBoard / W&B logging | High | Medium |
