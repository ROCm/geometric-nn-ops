# Contributing to geometric-nn-ops

We are enthusiastic about contributions to our code and documentation. Please
feel free to file issues where documentation or functionality is lacking or,
even better, volunteer to contribute to help close these gaps!

______________________________________________________________________

> **Security vulnerabilities** — do not open a public GitHub Issue. See [SECURITY.md](SECURITY.md) for the private reporting process.

______________________________________________________________________

## Developer policies

These policies apply to all forms of activity and engagement in this project.

### Governance

This project is covered by the
[ROCm Project Governance](https://github.com/ROCm/TheRock/blob/main/GOVERNANCE.md),
which also defines the code of conduct.

### Licensing

Code contributions to this project are covered under the terms of the
[LICENSE](LICENSE) file.

### Communication channels

Issue tracking, project planning, and code contributions are managed in GitHub.
We use an open-source toolchain so that workflows can be easily replicated in any fork.

______________________________________________________________________

## Development workflows

### Issue tracking

Before filing a new issue, search through
[existing issues](https://github.com/ROCm/geometric-nn-ops/issues) to avoid duplicates.

General guidelines:

- If your issue is already listed, upvote it and add a comment with reproduction details.
- When in doubt, file a new issue — we'll mark duplicates accordingly.
- Provide as much information as possible: command output, GPU model, ROCm (or CUDA)
  version, PyTorch, Triton and Python versions, and OS version. This significantly reduces
  triage time.
- Check your issue regularly — we may ask follow-up questions.

### Development setup

Install PyTorch and Triton for your platform first, then:

```bash
pip install -e ".[test,dev]" -e "./hipequivariance_ops_torch[test]"
pre-commit install
```

PyTorch's ROCm wheels ship Triton as `triton-rocm` (older ones as
`pytorch-triton-rocm`); with those, do not run that first command: pip does not
count either as `triton` and would install PyPI's `triton` over it, a build that
need not match your PyTorch. Instead install the two
packages with `pip install --no-deps -e . -e ./hipequivariance_ops_torch`,
then the tools from the extras: `pip install "pytest==9.0.3" "pytest-cov==6.*" ruff==0.16.3 pre-commit cuequivariance==0.9.1 cuequivariance-torch==0.9.1`.

`pre-commit` runs the same hooks as CI: ruff lint and format, actionlint, and
general file hygiene. Run them on demand with `pre-commit run --all-files`.

### Code style

Follow the [ROCm/TheRock style guides](https://github.com/ROCm/TheRock/tree/main/docs/development/style_guides):

- [GitHub Actions style guide](https://github.com/ROCm/TheRock/blob/main/docs/development/style_guides/github_actions_style_guide.md)
- [Python style guide](https://github.com/ROCm/TheRock/blob/main/docs/development/style_guides/python_style_guide.md)

Deviations in this repository:

- Python is formatted and linted with [ruff](https://docs.astral.sh/ruff/)
  (Black-compatible), configured in `pyproject.toml`, rather than with Black.
- Test files are named `test_*.py`.
- Every Python file carries, near the top, a copyright line naming Advanced
  Micro Devices, Inc. followed by the SPDX identifier of the [LICENSE](LICENSE),
  enforced by ruff's `CPY001` rule. Files copied from elsewhere keep their
  original notices above it.

### New feature or test development

Discussion about new features and test areas is welcome via:

- Filing a [GitHub issue](https://github.com/ROCm/geometric-nn-ops/issues)

### Pull Requests

All contributions should be submitted through a Pull Request (PR).

Before submitting a PR:

- Ensure the change addresses an existing issue or documented requirement.
- Run all applicable tests and validate the results.
- Update documentation as needed.
- Ensure all required GitHub Actions checks pass.
- Resolve review feedback before requesting approval.

PRs require:

- Approval from the appropriate CODEOWNERS.
- Successful completion of required status checks.
- Compliance with repository security and quality requirements.

PRs are squash-merged.

### Branching

- Do not commit directly to protected branches.
- Create a feature branch for all changes. Without write access, fork the
  repository and open the PR from a branch in your fork.
- Keep changes focused and atomic.
- Rebase or update branches regularly to minimize merge conflicts.

### Commit Messages

- Use clear and descriptive commit messages.

### Testing

Contributors are expected to:

- Execute relevant unit and integration tests.
- Verify that existing functionality is not impacted.
- Include test coverage for new functionality when applicable.
- Document any known limitations or test exclusions.

Run the tests with `pytest tests hipequivariance_ops_torch/tests`. Tests marked
`large` need datacenter-class GPU memory and are deselected by default; include
them with `-m ""`. Most tests skip without a GPU.

GPU CI runs on a single gfx1151 runner, for pushes to `main` and same-repository
pull requests while the repository variable `RUN_GPU_TESTS` is `true`. It never covers
the supported gfx942 and gfx950 architectures, so a change to kernel code must
report local GPU test results on both in the pull request, or ask a maintainer to
run them.

### Documentation

Changes that impact functionality, configuration, APIs, or user workflows should include corresponding documentation updates.

### Security Requirements

Contributors must not:

- Commit secrets, tokens, passwords, or credentials.
- Introduce vulnerable dependencies without justification.
- Bypass security controls or required security reviews.

All contributions may be subject to automated security scanning.
