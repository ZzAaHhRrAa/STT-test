# Speech-to-Text OOP Inference Pipeline

An enterprise-ready, automated pipeline built in Python using Object-Oriented Programming (OOP) principles. The framework executes offline Whisper-tiny batch inference, applies strict symmetric text normalization tailored to benchmark constraints, tracks runtime performance metrics per audio stream, and generates structured analytical evaluation reports.

---

## Key Features

- **Strict Offline Capabilities:** Configured natively to prevent runtime external lookups (`HF_HUB_OFFLINE=1`), making it safe for air-gapped or secure enterprise computing environments.
- **Symmetric Normalization Engine:** Eliminates artificial evaluation penalties by routing both predictions and ground-truth text targets through a mirror cleaning matrix (handling expansions, specific spelling discrepancies, and uniform punctuation stripping).
- **Granular Profiling & Robust Logging:** Monitors transcription runtime metrics down to individual audio streams, outputting continuous operational data via Python’s `logging` module while silencing internal framework noise.
- **Isolated Containment:** Bundled with a standardized Docker architecture featuring local file-system mounting to avoid heavy asset caching inside the image.

---

## Project Structure

```text
STT-test/
│
├── data/
│   └── audio_files/        # Downloaded local audio files (Git-ignored)
│
├── src/
│   ├── __init__.py
│   ├── main.py          # Entrypoint script executing the pipeline
│   └── pipeline.py      # Core OOP logic (Data ingestion, Inference, Cleaning Engine)
|
├── Dockerfile           # Minimal Python 3.10 runtime build file
├── .dockerignore        # Context filter for building light containers
├── .gitignore           # Version control rules excluding runtime tracking/artifacts
├── requirements.txt     # Python application standard dependencies
└── README.md            # Project technical documentation
```