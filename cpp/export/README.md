# LavaSR ONNX Export

Convert the PyTorch models to ONNX format for the C++ inference engine.

## Prerequisites

```bash
pip install torch torchaudio vocos onnx onnxruntime huggingface_hub einops
pip install -e ../..   # install LavaSR in editable mode
```

## 1. Download model weights (if not already cached)

```python
from huggingface_hub import snapshot_download
model_dir = snapshot_download("YatharthS/LavaSR")
print(model_dir)   # copy this path for the commands below
```

## 2. Export BWE model

```bash
python export_bwe.py <model_dir> ./models [--cutoff 7500] [--v1]
```

Outputs: `models/bwe.onnx` + `models/bwe_config.json`

The script tries **Mode A** (full pipeline including ISTFT and FastLRMerge in
the ONNX graph). If the ISTFT operator is not supported by your PyTorch/ONNX
version, it falls back to **Mode B** (backbone only; C++ handles ISTFT and
FastLRMerge). The `bwe_config.json` records which mode was selected so the
C++ runtime knows what to do.

## 3. Export denoiser model

```bash
python export_denoiser.py <model_dir>/denoiser/denoiser.bin ./models/denoiser.onnx
```

Outputs: `models/denoiser.onnx` + `models/denoiser_config.json`

The denoiser always uses split mode: STFT/ISTFT are performed in C++; the
ONNX model only processes the frequency-domain features.

## File layout expected by the C++ engine

```
models/
├── bwe.onnx
├── bwe_config.json
├── denoiser.onnx
└── denoiser_config.json
```

Pass `models/bwe.onnx` and `models/denoiser.onnx` to `lavasr_create()`.
