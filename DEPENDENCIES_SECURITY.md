# Dependency Security Updates

This file tracks known vulnerabilities in dependencies and recommended updates.

**Last Updated**: 2025-11-23

## Critical Updates Needed

### 1. cryptography (CRITICAL - Multiple CVEs)

**Current Version**: 41.0.7
**Recommended Version**: 43.0.1+

**Vulnerabilities**:
- **PYSEC-2024-225**: NULL pointer dereference in PKCS12 (Fix: 42.0.4+)
- **GHSA-3ww4-gg4f-jr7f**: TLS RSA key exchange vulnerability (Fix: 42.0.0+)
- **GHSA-9v9h-cgj8-h64p**: PKCS12 malformed file DoS (Fix: 42.0.2+)
- **GHSA-h4gh-qq45-vh27**: OpenSSL statically linked vulnerability (Fix: 43.0.1+)

**Update Command**:
```bash
pip install --upgrade cryptography>=43.0.1
```

### 2. pip (HIGH - Path Traversal)

**Current Version**: 24.0
**Recommended Version**: 25.3+

**Vulnerability**:
- **GHSA-4xh5-x5gv-qwph**: Path traversal in tarfile extraction allowing arbitrary file overwrite

**Update Command**:
```bash
pip install --upgrade pip>=25.3
```

### 3. setuptools (HIGH - Path Traversal)

**Current Version**: 68.1.2
**Recommended Version**: 78.1.1+

**Vulnerabilities**:
- **PYSEC-2025-49**: Path traversal in PackageIndex (Fix: 78.1.1+)
- **GHSA-cx63-2mw6-8hw5**: Remote code execution via package_index (Fix: 70.0+)

**Update Command**:
```bash
pip install --upgrade setuptools>=78.1.1
```

## Quick Fix All

To update all vulnerable packages at once:

```bash
pip install --upgrade pip>=25.3 setuptools>=78.1.1 cryptography>=43.0.1
```

## Verification

After updating, verify with:

```bash
pip-audit --desc
```

Expected output should show no vulnerabilities.

## CI/CD Integration

Our GitHub Actions workflow automatically checks for vulnerabilities using:
- **Bandit**: Code security linter
- **pip-audit**: PyPI package vulnerability scanner

Security reports are uploaded as artifacts for each CI run.

## Regular Maintenance

**Recommendation**: Run security audits regularly:

```bash
# Check for vulnerabilities
pip-audit

# Check for outdated packages
pip list --outdated

# Update all packages (with caution)
pip list --outdated --format=json | jq -r '.[] | .name' | xargs pip install --upgrade
```

## Dependency Pinning

After updating and testing, pin versions in requirements.txt:

```bash
pip freeze > requirements-lock.txt
```

## Notes

- **cryptography**: This is likely a transitive dependency (through PyTorch, requests, or other packages)
- **pip/setuptools**: System packages that should be updated globally
- Always test after updates to ensure compatibility

## Impact Assessment

| Package | Severity | Exploitability | Impact | Priority |
|---------|----------|----------------|--------|----------|
| cryptography | HIGH | Medium | RCE/DoS | **Immediate** |
| pip | HIGH | Low | File Overwrite | **Immediate** |
| setuptools | HIGH | Medium | RCE | **Immediate** |

## Action Items

- [ ] Update cryptography to 43.0.1+
- [ ] Update pip to 25.3+
- [ ] Update setuptools to 78.1.1+
- [ ] Run full test suite after updates
- [ ] Update CI to use updated versions
- [ ] Document any breaking changes

## References

- [pip security advisory GHSA-4xh5-x5gv-qwph](https://github.com/advisories/GHSA-4xh5-x5gv-qwph)
- [setuptools security advisory PYSEC-2025-49](https://osv.dev/vulnerability/PYSEC-2025-49)
- [cryptography security advisories](https://github.com/pyca/cryptography/security/advisories)
