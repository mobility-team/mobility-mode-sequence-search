# Releasing

## Initial Setup

These steps are usually done once per repository:

1. Configure this repository as a Trusted Publisher on PyPI.
2. Create a protected GitHub environment named `pypi`.
3. Ensure the release workflow in [.github/workflows/wheels.yml](/d:/dev/mobility-mode-sequence-search/.github/workflows/wheels.yml) is enabled.

## Per Release

For each package release:

1. Update the version in [pyproject.toml](/d:/dev/mobility-mode-sequence-search/pyproject.toml).
2. Commit the version change.
3. Create and push a version tag such as `v0.1.0`.
4. Let GitHub Actions build the wheels and source distribution.
5. Let the `publish` job upload the distributions to PyPI.
6. Verify the published files and metadata on PyPI.
