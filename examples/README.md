# DynWGraphs Examples

This directory contains example scripts demonstrating how to use the dynwgraphs package.

## Examples

### 01_static_weighted_network.py

Demonstrates basic usage of static weighted network models:
- Generating synthetic network data
- Estimating model parameters
- Evaluating model fit
- Visualizing results

**Run:**
```bash
python examples/01_static_weighted_network.py
```

### 02_dynamic_score_driven_network.py

Demonstrates score-driven dynamics for time-varying networks:
- Estimating dynamic parameters
- Filtering and forecasting
- Out-of-sample evaluation
- Visualizing parameter evolution

**Run:**
```bash
python examples/02_dynamic_score_driven_network.py
```

## Requirements

All examples require the dynwgraphs package and its dependencies:

```bash
pip install -e .
```

## Output

Each example generates:
- Console output with progress updates and results
- PNG figure files with visualizations

## Next Steps

After running these examples, check out:
- The test files in `tests/` for more detailed usage
- The main package documentation
- The module docstrings for API details
