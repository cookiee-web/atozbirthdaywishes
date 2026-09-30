# ISL model training

`train_model.py` trains two models from the datasets stored in this folder:

- `DATABASE ISL/ISL_Dataset`: static alphabet hand images, trained as `models/isl_alphabet.keras`.
- `Dataset of Common phrases in ISL/Dataset of Common phrases in Indian Sign Language.zip`: 41 labelled 30-frame holistic landmark sequences, trained as `models/isl_phrases.keras`.

## Run

From the `ASL` folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
py train_model.py
```

To retrain one model while keeping the other model's metrics, use `--only alphabet` or `--only phrases`.

The script writes label maps and validation metrics to `models/`. The phrase ZIP is extracted only into the generated `generated/phrase_dataset` directory. Generated models and extracted data should not be committed to source control.

The current `index.html` prototype still uses its geometric MediaPipe classifier. The exported Keras models are the trained artifacts; browser inference should be added as a separate step using TensorFlow.js conversion after validating these metrics.