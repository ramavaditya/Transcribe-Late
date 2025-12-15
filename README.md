## Indic TUI: Indian-accent transcription + translation (Ollama or IndicTrans2)

### Features
- Transcribe Indian-accent speech locally with Faster-Whisper
- Translate with local Ollama model or IndicTrans2 (Transformers)
- Simple TUI to pick audio, choose target language, and run

### Prerequisites
- Windows 10/11, PowerShell 7
- Python 3.10 or newer
- FFmpeg installed and on PATH (required for many audio formats)
- Optional: Ollama installed and running (`ollama serve`) if using Ollama translation

### Install
```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

### Run
```bash
python app.py
```

### Notes
- First run will download models (Faster-Whisper and/or IndicTrans2). This can be several GB.
- To use IndicTrans2, the model used is `ai4bharat/indictrans2-en-indic-1B`.
- To use Ollama, set the translator to "ollama" in the TUI and select a model like `llama3.1:8b`.
