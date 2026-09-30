# ISL Setu

ISL Setu is a prototype for Indian Sign Language recognition. It contains a browser interface and Python training code for two models:

- alphabet recognition from labelled hand images;
- common-phrase recognition from 30-frame MediaPipe landmark sequences.

The source repository intentionally excludes the datasets, virtual environment, generated extraction directory, and trained model binaries. Keep those assets locally or publish them through a dataset or model registry instead of Git.

## Project layout

```text
index.html       Browser prototype
train_model.py   Model training entry point
requirements.txt Python runtime dependencies
TRAINING.md      Dataset and training instructions
models/          Local model outputs (ignored by Git)
```

## Quick start

Create a Python environment and install the pinned dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

Place the datasets in the paths documented in [TRAINING.md](TRAINING.md), then train one or both models:

```powershell
py train_model.py --only alphabet
py train_model.py --only phrases
py train_model.py
```

The training command writes model files, label maps, and metrics to `models/`. These outputs are local artifacts and are excluded from source control.

## Browser prototype

Open `index.html` in a browser or serve the folder with a local static server. Camera access requires `localhost` or HTTPS in most browsers.

The browser prototype currently uses its client-side geometric classifier. The Keras models are training outputs and are not automatically loaded by the page.

## Development notes

This is an experimental accessibility project. Recognition results should be treated as assistive suggestions, not as a replacement for a qualified interpreter or a user’s preferred communication method.

Before opening a pull request, check that Python syntax is valid and that no datasets, virtual environments, credentials, or generated binaries are included:

```powershell
py -m compileall train_model.py
git status --short
```