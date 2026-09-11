# SchemaGAN — PyTorch implementation

## Description

This folder contains a PyTorch implementation of [SchemaGAN](https://github.com/fabcamo/schemaGAN),
the conditional Generative Adversarial Network that takes
Cone Penetration Test (CPT) data as a conditional input and generates subsoil schematisations. It reproduces
the architecture and the training objective of the original TensorFlow/Keras implementation.

The original work is described in:

**Campos Montero, F.A., Zuada Coelho, B., Smyrniou, E., Taormina, R., & Vardon, P.J. (2025)**
*SchemaGAN: A conditional Generative Adversarial Network for geotechnical subsurface schematisation*
Computers and Geotechnics, 183, 107177
[https://doi.org/10.1016/j.compgeo.2025.107177](https://doi.org/10.1016/j.compgeo.2025.107177)

If you use SchemaGAN in your work, please cite this paper.



## Installation

It is recommended to use a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The PyTorch implementation needs only `torch`, `numpy`, `matplotlib`, `pyyaml` and `pytest`.
A CUDA device is picked up automatically and can be overridden with `train.device` in the settings file.


## Layout

| Path | Purpose |
|------|---------|
| [schemagan.py](schemagan.py) | The `SchemaGAN` class: `train`, `validate`, `test`, `predict`, `save`/`load` |
| [config.py](config.py) | `DataConfig`, `ModelConfig`, `OptimConfig`, `TrainConfig`, `SchemaGANConfig` |
| [cli.py](cli.py) | Reads the YAML settings file used by the three scripts |
| [../configs/default.yaml](../configs/default.yaml) | All settings of a run |
| [data.py](data.py) | CSV reading/writing, CPT simulation, normalisation, `CrossSectionDataset` |
| [models/](models) | `UNetGenerator`, `PatchDiscriminator`, `ConvAutoencoder`, `AutoencoderGenerator` |
| [metrics.py](metrics.py) | MAE, MSE and RMSE, per sample and aggregated |
| [visualize.py](visualize.py) | Comparison figures, loss curves and error histograms |
| [../training_schemaGAN_torch.py](../training_schemaGAN_torch.py) | Train a model |
| [../validation_schemaGAN_torch.py](../validation_schemaGAN_torch.py) | Score one or more checkpoints |
| [../inference_schemaGAN_torch.py](../inference_schemaGAN_torch.py) | Generate cross-sections |

## Usage

### 1. Prepare the images

The subsoil schematisations used in this work were generated with
[GeoSchemaGen](https://github.com/fabcamo/GeoSchemaGen); the training, validation and test datasets are
available on [Zenodo](https://zenodo.org/records/13143431/files/data.zip).

The images must be 2D CSV files with dimensions (512, 32), in the same long format as the original
implementation:

| index | x | z | IC value |
|-------|---|---|----------|
| 0 | 0 | 0 | 2.7 |
| 1 | 0 | 1 | 1.5 |
| 2 | 0 | 2 | 1.7 |
| .... | | | |
| 16383 | 511 | 31 | 2.6 |

Rows may appear in any order: the grid is filled through the `x` and `z` columns.

### 2. Configure the run

Every script reads one YAML file, [configs/default.yaml](configs/default.yaml) unless another path is
given. Copy it and edit the values; omitted keys fall back to the dataclass defaults.

```yaml
data:
  miss_rate: 0.99                  # keep ~1% of the columns as CPTs
  min_distance: 51                 # minimum spacing between two CPTs

model:
  generator_filters: 64
  discriminator_filters: 64

optim:
  lambda_l1: 100.0                 # weight of the reconstruction term

train:
  epochs: 10
  batch_size: 1
  device: auto                     # auto, cpu, cuda or cuda:N
  seed: 42
  checkpoint_every: 1              # 0 switches the periodic checkpoints off
  sample_every: 1                  # 0 switches the periodic figures off

training:
  data_dir: synthetic_data/512x32/train
  val_dir: synthetic_data/512x32/validation
  output_dir: results/torch_run
  resample_mask: false             # new CPT layout on every access
```

The `data`, `model`, `optim` and `train` sections describe the model and are stored in every checkpoint;
the `training`, `validation` and `inference` sections hold the paths and options of the matching script.

### 3. Train

```bash
python training_schemaGAN_torch.py configs/default.yaml
```

The run directory receives:

```
results/torch_run/
├── config.json                      # exact settings of the run
├── final_model.pt                   # weights, optimiser state and history
├── history.png                      # d_loss / g_loss / g_l1 curves
├── history_per_iteration.csv
├── history_per_epoch.csv
├── validation_summary.json          # only with training.val_dir
├── checkpoints/schemagan_epoch_000001.pt
└── samples/epoch_000001_000.png
```

### 4. Validate

```bash
python validation_schemaGAN_torch.py configs/default.yaml
```

`validation.checkpoint` accepts a single `.pt` file or a directory, which makes it easy to watch the error
evolve over the epochs. For every checkpoint the script writes `errors_<name>.csv` (MAE, MSE and RMSE per
cross-section), `mae_histogram_<name>.png` and a few comparison figures, plus a combined `summary.csv`.

### 5. Inference

```bash
python inference_schemaGAN_torch.py configs/default.yaml
```

`inference.simulate_cpt` masks complete cross-sections to obtain the sparse input; switch it off when the
CSV files already hold CPT-like data with zeros at the unmeasured pixels. Each input produces
`<name>_generated.csv` in the same long format and a `<name>_generated.png` figure (`inference.plots: false`
skips the figures). The image geometry and the IC range are read from the checkpoint.

Example of the results:
![Schematisations](../static/plot_acc_000000.png)

## Using the library directly

Training and scoring a model from Python:

```python
from schemaGAN_torch import CrossSectionDataset, SchemaGAN, SchemaGANConfig

config = SchemaGANConfig()
config.train.epochs = 10
config.data.miss_rate = 0.99        # keep ~1% of the columns as CPTs
config.data.min_distance = 51       # minimum spacing between two CPTs

train_data = CrossSectionDataset("synthetic_data/512x32/train", config.data, seed=42)
val_data = CrossSectionDataset("synthetic_data/512x32/validation", config.data, seed=42)

model = SchemaGAN(config)
model.train(train_data, val_data=val_data, output_dir="results/torch_run")

result = model.validate(val_data)    # per cross-section errors, in IC units
print(result.summary)                # {'mae': ..., 'mse': ..., 'rmse': ...}
result.to_csv("results/torch_run/errors.csv")
```

Generating a schematisation from a stored checkpoint:

```python
import numpy as np

from schemaGAN_torch import SchemaGAN
from schemaGAN_torch.data import apply_mask, cpt_like_mask, normalize_ic, read_cross_section_csv
from schemaGAN_torch.visualize import plot_cross_sections

model = SchemaGAN.load("results/torch_run/final_model.pt")
data = model.config.data

target = read_cross_section_csv("example_schematisations/cs_3.csv", data.image_height, data.image_width)
mask = cpt_like_mask(
    data.image_height, data.image_width, data.miss_rate, data.min_distance, np.random.default_rng(42)
)
source = apply_mask(target, mask)

generated = model.predict(normalize_ic(source, data.min_ic, data.max_ic), denormalize=True)
plot_cross_sections(source, generated[0, 0], target, "results/cs_3.png")
```

`predict` accepts a `(32, 512)` grid, a `(n, 32, 512)` batch or a `(n, 1, 32, 512)` tensor and always
returns a CPU tensor of shape `(n, 1, 32, 512)`.

---

## Architecture

Both networks follow the original design. Because a cross-section is 32 px deep but 512 px wide, half of the
blocks downsample the horizontal axis only, with a `(1, 2)` stride.

**Generator** — U-Net with skip connections, `(32, 512, 1) → (32, 512, 1)`:

| Stage | Blocks | Stride | Output |
|-------|--------|--------|--------|
| Encoder | C64, C128, C256, C512 | (2, 2) | 2×32×512 |
| Encoder | C512 × 4 | (1, 2) | 2×2×512 |
| Bottleneck | C512, ReLU | (2, 2) | 1×1×512 |
| Decoder | CD512 (dropout 0.5) × 4 | (2, 2), then (1, 2) | 2×16×1024 |
| Decoder | C512, C256, C128, C64 | (1, 2), then (2, 2) | 16×256×128 |
| Output | transposed conv, tanh | (2, 2) | 32×512×1 |

**Discriminator** — PatchGAN over the concatenated (source, target) pair, producing a 16×16 patch map:
C64 → C128 → C256 → C512 → C512 → C512 → 1.

**Objective** — `BCE + 100 · L1` for the generator, `0.5 · (BCE_real + BCE_fake)` for the discriminator,
both optimised with Adam (`lr = 2e-4`, `β₁ = 0.5`).

---
<!--
## Connecting an auto-encoder

`SchemaGAN` accepts any `nn.Module` as generator or discriminator, and `UNetGenerator` exposes `encode`,
`decode` and `fuse_latent`, so an auto-encoder can be attached without touching the training loop:

```python
from schemaGAN_torch import SchemaGAN, SchemaGANConfig
from schemaGAN_torch.models import AutoencoderGenerator, ConvAutoencoder, UNetGenerator

config = SchemaGANConfig()
config.model.latent_channels = 64          # width of the code fused into the bottleneck

generator = AutoencoderGenerator(
    ConvAutoencoder(latent_channels=64),
    UNetGenerator(latent_channels=64),
    mode="latent",                          # or "preprocess"
    freeze_autoencoder=True,                # keep a pre-trained auto-encoder fixed
)

model = SchemaGAN(config, generator=generator)
```

`mode="latent"` feeds the auto-encoder code into the generator bottleneck, while `mode="preprocess"` runs
the generator on the auto-encoder reconstruction. When reloading such a model, pass the same module to
`SchemaGAN.load(path, generator=...)`, since only the weights are stored. -->



## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.