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