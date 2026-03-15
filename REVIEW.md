# Repository Review: dynwgraphs

## Overview

`dynwgraphs` is a PyTorch-based scientific computing library for modeling dynamic weighted sparse graphs using zero-augmented distributions. It's in pre-alpha (v0.1.0), scaffolded from `cookiecutter-pypackage`.

---

## Critical Issues

### 1. Bug in `setup.py` — Missing comma in dependencies

```python
# Line 13: missing comma causes string concatenation
requirements = ['Click>=7.0', 'numpy>=1.19', 'pandas>=1.2', 'scipy>=1.6' 'torch']
#                                                                      ^ missing comma
# This produces 'scipy>=1.6torch' instead of two separate entries
```

**Fix:** Add the missing comma: `'scipy>=1.6', 'torch'`

### 2. Bare `raise` statements without exception types

Throughout `dirGraphs1_dynNets.py`, there are bare `raise` statements (lines 100, 102, 109, 113, 146, 149, 206, 278, 281, 285, 286, etc.) that will raise `RuntimeError` with no message, making debugging extremely difficult.

**Fix:** Replace with descriptive exceptions, e.g.:
```python
raise ValueError("X_t is provided but beta is None — both must be present or absent")
```

### 3. String used as exception (`raise "Not ready"`, line 146)

`raise "Not ready"` is a `TypeError` in Python 3 — it doesn't raise a string as an error. This is a silent bug.

**Fix:** `raise NotImplementedError("Symmetric phi (N-dimensional) not yet implemented")`

### 4. Undefined variable bug in `exp_Y_T` (lines 183-193)

```python
def exp_Y_T(self, phi_T, beta_T, X_T):
    for t in range(T):
        if beta_T is None:
            beta_t = None
        if X_T is None:
            X_t = None
        # BUG: if beta_T is NOT None, beta_t is never assigned
        # BUG: if X_T is NOT None, X_t is never assigned
        Y_T[:, :, t] = self.exp_Y(phi_T[:, t], beta_t, X_t)
```

The `if` blocks should be `if/else`, extracting `beta_T[:, t]` and `X_T[:, :, :, t]` in the else branches.

### 5. Missing dependencies in `setup.py`

`scikit-learn`, `statsmodels`, and `matplotlib` are imported in core code but not listed in `install_requires`. Users will get `ImportError` on a clean install.

---

## Architectural Issues

### 6. Monolithic 2,800-line core module

`dirGraphs1_dynNets.py` contains 7+ classes and all model logic in a single file. This makes navigation, testing, and maintenance difficult.

**Recommendation:** Split into:
- `base.py` — `dirGraphs_funs` base class
- `sequence.py` — `dirGraphs_sequence_ss` sequential logic
- `state_dependent.py` — `dirGraphs_SD` dynamic/state-dependent logic
- `weighted.py` — `dirSpW1_*` weighted model specializations
- `binary.py` — `dirBin1_*` binary model specializations
- `likelihoods.py` — likelihood computation functions

### 7. Empty placeholder packages

`tobit/` and `adahessian/` directories exist with empty `__init__.py` files. These add confusion without value.

**Recommendation:** Remove until they have actual content, or add a clear roadmap in a `TODO.md`.

### 8. Class hierarchy is deep and unclear

The inheritance chain is: `dirGraphs_funs` -> `dirGraphs_sequence_ss` -> `dirGraphs_SD` -> `dirSpW1_sequence_ss` -> `dirSpW1_SD`. The responsibilities at each level are not well-delineated, and method overrides are hard to trace.

**Recommendation:** Consider composition over inheritance, or at minimum add clear docstrings explaining what each class layer adds.

---

## Code Quality Issues

### 9. No type hints anywhere

The entire codebase lacks type annotations. For a mathematical library with complex tensor operations, this is particularly problematic — it's hard to know what shape/type a parameter should be.

**Recommendation:** Add type hints progressively, starting with public API methods:
```python
def estimate_ss_t(self, Y_t: torch.Tensor, phi_0: Optional[torch.Tensor] = None, ...) -> Tuple[torch.Tensor, dict]:
```

### 10. Inconsistent naming conventions

- Classes use mixed styles: `dirGraphs_funs`, `dirSpW1_sequence_ss`, `dirBin1_SD`
- Methods mix camelCase and snake_case: `putZeroDiag` vs `start_phi_from_obs`
- Abbreviations are opaque: `ss`, `SD`, `W1`, `IO`, `un`, `dist_par_un_t`

**Recommendation:** Adopt consistent snake_case for functions/methods and CamelCase for classes, with a glossary in the README for domain abbreviations.

### 11. Print statements used for debugging instead of logging

Tests and some model code use `print()` for output. The `logger` is defined but underused.

**Recommendation:** Replace `print()` calls with `logger.info()` / `logger.debug()`.

---

## Testing Issues

### 12. Tests are integration-heavy, no unit tests

All tests are end-to-end estimation workflows. There are no unit tests for individual methods like `regr_product`, `identify_io_par_to_be_sum`, `splitVec`, etc.

**Recommendation:** Add unit tests for utility functions and individual class methods. These run faster and pinpoint failures precisely.

### 13. Tests use relative paths for test data

```python
test_data = np.load("../tests/test_data/dir_w_test_data.npz")
```

This breaks when tests are run from different working directories.

**Recommendation:** Use `pathlib.Path(__file__).parent` for robust path resolution.

### 14. No test isolation — matplotlib side effects

Tests create matplotlib figures without `plt.close()`, which can leak memory and cause issues in CI.

**Recommendation:** Use `@pytest.fixture` for setup/teardown and `plt.close('all')` in cleanup.

---

## Infrastructure Issues

### 15. Outdated CI/CD — Travis CI with Python 3.6/3.7/3.8

- Travis CI is largely deprecated for open-source projects
- Python 3.6 and 3.7 are EOL
- PyTorch no longer supports Python 3.6/3.7
- Deploy password is a placeholder

**Recommendation:** Migrate to GitHub Actions targeting Python 3.9-3.12. Example:
```yaml
# .github/workflows/test.yml
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11', '3.12']
```

### 16. Pinned dev dependency versions are very old

`requirements_dev.txt` pins old versions (e.g., `pip==19.2.3`, `flake8==3.7.8`, `Sphinx==1.8.5`).

**Recommendation:** Update to current versions, or use `>=` constraints for dev tools.

### 17. No `pyproject.toml`

The project still uses the legacy `setup.py` + `setup.cfg` pattern.

**Recommendation:** Migrate to `pyproject.toml` (PEP 621) for modern Python packaging.

---

## Documentation Issues

### 18. README is a stub

The Features section just says "TODO". There's no usage example, installation instructions, or description of the models.

**Recommendation:** Add at minimum:
- What the package does (1-2 paragraphs)
- Installation instructions
- A minimal usage example
- Links to any related papers

### 19. No API documentation

Sphinx is set up but contains only stubs. Most functions lack docstrings.

**Recommendation:** Add NumPy-style docstrings to all public classes and methods, then generate API docs.

### 20. No examples or tutorials

For a scientific library, examples are essential for adoption.

**Recommendation:** Add a `examples/` directory with Jupyter notebooks demonstrating key workflows.

---

## Prioritized Improvement Plan

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| **P0** | Fix `setup.py` comma bug (#1) | 5 min | Broken installs |
| **P0** | Fix `exp_Y_T` undefined variable bug (#4) | 10 min | Runtime crash |
| **P0** | Fix bare `raise` and string raise (#2, #3) | 30 min | Debuggability |
| **P0** | Add missing deps to `setup.py` (#5) | 5 min | Broken installs |
| **P1** | Add proper exception messages | 1 hr | Debuggability |
| **P1** | Write a real README with examples (#18) | 2 hr | Adoption |
| **P1** | Migrate CI to GitHub Actions (#15) | 1 hr | CI actually works |
| **P1** | Update Python version targets to 3.9+ (#15) | 30 min | Compatibility |
| **P2** | Add type hints to public API (#9) | 4 hr | Maintainability |
| **P2** | Split monolithic module (#6) | 4 hr | Maintainability |
| **P2** | Add unit tests (#12) | 4 hr | Reliability |
| **P2** | Migrate to `pyproject.toml` (#17) | 1 hr | Modern packaging |
| **P3** | Fix naming conventions (#10) | 2 hr | Readability |
| **P3** | Add docstrings and API docs (#19) | 4 hr | Adoption |
| **P3** | Add examples/tutorials (#20) | 4 hr | Adoption |
| **P3** | Remove empty packages (#7) | 5 min | Cleanliness |
