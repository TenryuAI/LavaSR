"""
Export ULUNAS denoiser inner network to ONNX (split mode).

The STFT and ISTFT are handled by C++; the ONNX model receives the
pre-computed STFT spectrum and returns the enhanced spectrum.

ONNX model I/O:
  Input:  stft_2ch  (1, 2, T_frames, 257)  – [real, imag] of onesided STFT
  Output: stft_enh  (1, 2, T_frames, 257)  – enhanced [real, imag]

STFT parameters (fixed for ULUNAS):
  n_fft=512, hop_length=256, win_length=512, Hann window, onesided

Usage:
    python export_denoiser.py <denoiser_bin> <output_path>

  <denoiser_bin>  : path to denoiser/denoiser.bin
  <output_path>   : path for denoiser.onnx (config saved alongside)
"""

import argparse
import json
import os
import sys

# Fix Windows cp932/gbk encoding for torch.onnx emoji output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from LavaSR.denoiser.ulunas import ULUNAS


# ---------------------------------------------------------------------------
# Inner network wrapper: takes STFT 2ch, returns enhanced STFT 2ch
# ---------------------------------------------------------------------------

class ULUNASInner(nn.Module):
    """
    ULUNAS without STFT/ISTFT.

    forward(stft_2ch) where stft_2ch: (B, 2, T_frames, F=257)
    returns:                          (B, 2, T_frames, F=257)

    The caller (C++) is responsible for:
      pre : STFT(wav_16k) → stft_2ch
      post: stft_enh → ISTFT → wav_16k_enhanced
    """

    def __init__(self, ulunas: ULUNAS):
        super().__init__()
        self.erb     = ulunas.erb
        self.encoder = ulunas.encoder
        self.dpgrnn  = ulunas.dpgrnn
        self.decoder = ulunas.decoder

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        # spec: (B, 2, T_frames, F)
        # --- log magnitude feature ---
        feat = torch.log10(torch.norm(spec, dim=1, keepdim=True).clamp(1e-12))
        # --- ERB compression ---
        feat = self.erb.bm(feat)
        # --- Encoder ---
        feat, en_outs = self.encoder(feat)
        # --- Dual-path RNN ---
        feat = self.dpgrnn(feat)
        # --- Decoder → mask ---
        m_feat = self.decoder(feat, en_outs)
        m = self.erb.bs(m_feat)       # (B, 1, T_frames, F) sigmoid mask
        # --- Apply mask ---
        return spec * m              # (B, 2, T_frames, F)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export(denoiser_bin: str, output_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"[Denoiser] Loading weights from {denoiser_bin}")
    ulunas = ULUNAS().eval()
    state = torch.load(denoiser_bin, map_location="cpu")
    ulunas.load_state_dict(state)

    inner = ULUNASInner(ulunas).eval()

    # dummy: 1 batch, 2 ch (re/im), 100 frames, 257 bins
    dummy = torch.randn(1, 2, 100, 257)

    print("[Denoiser] Exporting to ONNX (opset 17)…")
    # PyTorch 2.5+ defaults to the dynamo exporter which conflicts with
    # dynamic_axes on RNN-heavy models.  Force the legacy trace-based exporter
    # via dynamo=False so that dynamic_axes work as expected.
    try:
        torch.onnx.export(
            inner, (dummy,), output_path,
            input_names=["stft_2ch"],
            output_names=["stft_enh"],
            dynamic_axes={"stft_2ch": {2: "T_frames"}, "stft_enh": {2: "T_frames"}},
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    except TypeError:
        # dynamo= kwarg not available in this PyTorch build – use older API
        torch.onnx.export(
            inner, (dummy,), output_path,
            input_names=["stft_2ch"],
            output_names=["stft_enh"],
            dynamic_axes={"stft_2ch": {2: "T_frames"}, "stft_enh": {2: "T_frames"}},
            opset_version=17,
            do_constant_folding=True,
        )
    print(f"[Denoiser] Export OK → {output_path}")

    # ---- Write config (STFT params for C++ pre/post processing) ----------
    config = {
        "n_fft":      ulunas.n_fft,
        "hop_length": ulunas.hop_len,
        "win_length": ulunas.win_len,
        "sample_rate": 16000,
        "n_bins":     ulunas.n_fft // 2 + 1,
    }
    cfg_path = os.path.splitext(output_path)[0] + "_config.json"
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[Denoiser] Config saved → {cfg_path}")
    print(json.dumps(config, indent=2))

    # ---- ORT validation --------------------------------------------------
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(output_path,
                                    providers=["CPUExecutionProvider"])
        out = sess.run(None, {sess.get_inputs()[0].name: dummy.numpy()})
        print(f"[Denoiser] ORT validation OK, output shape: {out[0].shape}")
    except ImportError:
        print("[Denoiser] onnxruntime not installed – skipping ORT validation")
    except Exception as e:
        print(f"[Denoiser] ORT validation warning: {e}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Export ULUNAS denoiser to ONNX")
    p.add_argument("denoiser_bin",  help="Path to denoiser/denoiser.bin")
    p.add_argument("output_path",   help="Output path for denoiser.onnx")
    args = p.parse_args()
    export(args.denoiser_bin, args.output_path)
