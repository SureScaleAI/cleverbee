# Changelog

## [v0.1.2] - 2024-04-29

### Added
- Seamless update workflow: `run.sh` now keeps you up to date with the latest changes on the `main` branch, with prompts to pull and update dependencies if needed. No more detached HEAD states!

### Changed
- Upgraded to the latest PyMuPDF (now imported as `pymupdf` instead of `fitz`).
- `requirements.txt` now pins `pymupdf==1.25.5` for compatibility and future-proofing.

### How to update
Just run `./run.sh` as usual and follow the prompts! 