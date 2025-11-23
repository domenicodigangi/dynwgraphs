# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report security vulnerabilities by email to:
- **Email**: digangidomenico@gmail.com
- **Subject**: [SECURITY] dynwgraphs vulnerability report

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

You should receive a response within 48 hours. We will work with you to understand and address the issue promptly.

## Security Best Practices for Users

### 1. Pickle Files - Critical Security Risk

**WARNING**: The `load_par()` method uses Python's pickle module, which can execute arbitrary code.

**Safe Usage**:
```python
# ONLY load pickle files from trusted sources
model.load_par("parameters.pkl", trust_source=True)
```

**Never**:
- Load pickle files from untrusted sources
- Load pickle files downloaded from the internet without verification
- Load pickle files from user uploads without validation

**Alternative**: Consider saving/loading parameters in safer formats like JSON or HDF5 for untrusted data.

### 2. Input Validation

The package now uses `ast.literal_eval()` instead of `eval()` for parsing string inputs. However:

- Always validate user inputs before passing to the library
- Use type checking for critical parameters
- Sanitize file paths to prevent path traversal attacks

### 3. Dependencies

Keep dependencies up to date:
```bash
pip install --upgrade dynwgraphs
pip list --outdated
```

### 4. Secure File Operations

When saving/loading model files:
```python
import os

# Validate and sanitize file paths
file_path = os.path.normpath(user_provided_path)
if not file_path.startswith(safe_directory):
    raise ValueError("Invalid file path")
```

### 5. Data Files

When loading numpy `.npz` files:
```python
# Our code now disables pickle by default
# If you load data manually, always use:
data = np.load(file_path, allow_pickle=False)
```

## Security Features

###Recent Security Improvements (Version 0.1.0+)

1. **Code Injection Prevention**
   - Replaced `eval()` with `ast.literal_eval()` in parameter parsing
   - Only safe literal evaluation is now allowed

2. **Pickle Security**
   - Added `trust_source` parameter to `load_par()` method
   - Explicit confirmation required to load pickle files
   - Added file path validation and error handling
   - Warning logs when loading pickle files

3. **Numpy Load Security**
   - Set `allow_pickle=False` in test fixtures
   - Prevents arbitrary code execution from .npz files

4. **Input Validation**
   - Type checking on all user inputs
   - Path normalization to prevent directory traversal
   - Clear error messages for invalid inputs

## Known Security Considerations

### 1. Pickle Module Usage

The package uses Python's `pickle` module for parameter serialization. This is inherently insecure when loading untrusted files.

**Mitigation**:
- `trust_source` parameter must be explicitly set to `True`
- File path validation
- Logging warnings
- Documentation of risks

**Future Plans**:
- Implement safer alternatives (JSON, HDF5, safetensors)
- Provide migration tools

### 2. External Dependencies

The package depends on several external libraries. Monitor for security updates:
- PyTorch
- NumPy
- Pandas
- SciPy
- scikit-learn
- statsmodels

### 3. Optional Submodules

Two git submodules are optional (Adahessian, Tobit). These are not security-audited by this project.

## Vulnerability History

### Version 0.1.0 (2025)

**Fixed Vulnerabilities**:

1. **CVE-PENDING-001: Code Injection via eval()** (CRITICAL)
   - **Impact**: Arbitrary code execution through string parameters
   - **Affected**: `tv_flag_list_from_input()`, `tv_flag_bool_from_input()`
   - **Fix**: Replaced `eval()` with `ast.literal_eval()`
   - **Credit**: Internal security audit

2. **CVE-PENDING-002: Insecure Deserialization** (HIGH)
   - **Impact**: Arbitrary code execution when loading pickle files
   - **Affected**: `load_par()` method
   - **Fix**: Added `trust_source` parameter and validation
   - **Credit**: Internal security audit

3. **CVE-PENDING-003: Numpy Pickle Loading** (HIGH)
   - **Impact**: Arbitrary code execution from .npz files
   - **Affected**: Test fixtures
   - **Fix**: Set `allow_pickle=False`
   - **Credit**: Internal security audit

## Security Checklist for Contributors

Before submitting code:

- [ ] No use of `eval()` or `exec()`
- [ ] No use of `pickle.load()` without security warnings
- [ ] No use of `np.load()` without `allow_pickle=False`
- [ ] All file operations validate and normalize paths
- [ ] User inputs are validated and sanitized
- [ ] No hardcoded secrets or credentials
- [ ] Dependencies are pinned to secure versions
- [ ] Security implications documented in docstrings

## Security Tools

We use these tools for security:

1. **Bandit** - Python security linter
2. **Safety** - Dependency vulnerability checker
3. **pip-audit** - PyPI package auditing

Run security checks:
```bash
pip install bandit safety pip-audit
bandit -r dynwgraphs/
safety check
pip-audit
```

## Responsible Disclosure

We follow responsible disclosure practices:

1. Report received and acknowledged (48 hours)
2. Issue investigation and validation (1-2 weeks)
3. Fix development and testing (1-4 weeks)
4. Security advisory published
5. CVE requested if applicable
6. Fix released with security notes

## Contact

For security concerns:
- **Email**: digangidomenico@gmail.com
- **Subject**: [SECURITY] dynwgraphs

For general issues:
- **GitHub Issues**: https://github.com/domenicodigangi/dynwgraphs/issues

## Acknowledgments

We thank the security researchers and users who help keep dynwgraphs secure.

---

**Last Updated**: 2025-01-23
