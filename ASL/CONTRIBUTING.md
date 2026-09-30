# Contributing

## Local setup

Follow the setup instructions in [README.md](README.md) and [TRAINING.md](TRAINING.md). Keep large datasets and generated model files outside commits; the repository `.gitignore` is configured for the expected local paths.

## Pull requests

- Keep changes focused and explain user-visible or training-impacting behavior.
- Do not commit `.venv`, datasets, generated extraction output, model binaries, secrets, or personal editor settings.
- Run `py -m compileall train_model.py` before submitting Python changes.
- Update the documentation when commands, paths, or supported workflows change.