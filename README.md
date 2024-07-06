# Refiners Model Explorer

A Model Explorer extension to visualize and explore Refiners-based model.

## Installation

```bash
pip install git+https://github.com/finegrain-ai/model_explorer_refiners.git
```

## Usage

```python
from refiners.foundationals.dinov2 import DINOv2_small_reg
from model_explorer_refiners import RefinersAdapter

model = DINOv2_small_reg()
RefinersAdapter.visualize(model)

```
