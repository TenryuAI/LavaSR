"""
Export LavaBWE (Vocos backbone + FastLRMerge) to ONNX.

Two export modes are attempted:
  Mode A "full"  : wav_48k (1, T) → enhanced_48k (1, T)
                   ONNX graph includes ISTFT and FastLRMerge.
  Mode B "split" : wav_48k (1, T) → raw_head_out (1, n_fft+2, T_frames)
                   C++ applies ISTFT + FastLRMerge.

Mode A is tried first; if torch.onnx.export fails (e.g. ISTFT not supported),
we fall back to Mode B automatically.

Usage:
    python export_bwe.py <model_dir> <output_dir> [--cutoff 7500] [--v2]

  <model_dir>  : local HuggingFace snapshot root (e.g. ~/.cache/huggingface/...)
  <output_dir> : destination directory for bwe.onnx and bwe_config.json
"""

import argparse
import io
import sys

# Fix Windows console encoding (cp932/gbk) that cannot represent ✅/❌ emojis
# printed internally by torch.onnx – must be done before any torch import.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from LavaSR.enhancer.enhancer import LavaBWE


# ---------------------------------------------------------------------------
# ONNX-friendly FastLRMerge (element-wise ops only, no integer indexing)
# ---------------------------------------------------------------------------

class _FastLRMergeONNX(nn.Module):
    """Pure tensor-op implementation of FastLRMerge compatible with ONNX export."""

    def __init__(self, cutoff_hz: int, sample_rate: int, transition_bins: int):
        super().__init__()
        self.cutoff_fraction = float(cutoff_hz) / (sample_rate / 2.0)
        self.half_tw = float(transition_bins) / 2.0

    def forward(self, pred: torch.Tensor, orig: torch.Tensor) -> torch.Tensor:
        # pred, orig: (1, T) float32
        spec_p = torch.fft.rfft(pred, dim=-1)
        spec_o = torch.fft.rfft(orig, dim=-1)

        n_bins = spec_p.shape[-1]
        # bins as float for pure arithmetic (avoids Python int → constant baking)
        bins = torch.arange(n_bins, dtype=torch.float32, device=pred.device)

        cutoff_bin = self.cutoff_fraction * float(n_bins)
        t = (bins - cutoff_bin) / self.half_tw           # -1..1 in transition
        t_c = t.clamp(-1.0, 1.0)
        t01 = (t_c + 1.0) * 0.5                          # 0..1
        mask_real = 3.0 * t01 ** 2 - 2.0 * t01 ** 3     # smoothstep
        mask = mask_real.to(torch.complex64)

        merged = spec_o + (spec_p - spec_o) * mask
        return torch.fft.irfft(merged, n=pred.shape[-1], dim=-1)


# ---------------------------------------------------------------------------
# Mode A: full pipeline (includes custom ISTFT head + FastLRMerge)
# ---------------------------------------------------------------------------

class BWEFull(nn.Module):
    def __init__(self, bwe: LavaBWE, cutoff_hz: int, sample_rate: int,
                 transition_bins: int):
        super().__init__()
        self.feature_extractor = bwe.bwe_model.feature_extractor
        self.backbone = bwe.bwe_model.backbone
        self.head = bwe.bwe_model.head           # already monkey-patched
        self.merge = _FastLRMergeONNX(cutoff_hz, sample_rate, transition_bins)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:  # (1, T) → (1, T)
        feat = self.feature_extractor(wav)
        feat = self.backbone(feat)
        pred = self.head(feat)                   # (1, T') via custom_forward

        T = wav.shape[-1]
        T_pred = pred.shape[-1]
        a = pred[:, :T].float()
        b = wav[:, :T_pred].float()
        return self.merge(a, b)


# ---------------------------------------------------------------------------
# Mode B: backbone + head.out only (C++ does ISTFT + FastLRMerge)
# ---------------------------------------------------------------------------

class BWESplit(nn.Module):
    def __init__(self, bwe: LavaBWE):
        super().__init__()
        self.feature_extractor = bwe.bwe_model.feature_extractor
        self.backbone = bwe.bwe_model.backbone
        # head.out maps backbone dim → (n_fft + 2) channels (mag + phase)
        self.head_out = bwe.bwe_model.head.out

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # returns (1, n_fft+2, T_frames) – caller does exp/ISTFT/merge
        feat = self.feature_extractor(wav)
        feat = self.backbone(feat)
        out = self.head_out(feat)          # (1, T_frames, n_fft+2)
        return out.transpose(1, 2)        # (1, n_fft+2, T_frames)


# ---------------------------------------------------------------------------
# Helpers to extract ISTFT params from the Vocos head
# ---------------------------------------------------------------------------

def _get_istft_params(bwe: LavaBWE) -> dict:
    head = bwe.bwe_model.head
    istft_mod = getattr(head, "istft", None)
    if istft_mod is None:
        return {"n_fft": 1024, "hop_length": 256, "win_length": 1024}
    return {
        "n_fft":      getattr(istft_mod, "n_fft",      1024),
        "hop_length": getattr(istft_mod, "hop_length",  256),
        "win_length": getattr(istft_mod, "win_length", 1024),
    }


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

def export(model_dir: str, output_dir: str, cutoff_hz: int = 7500,
           use_v2: bool = True, sample_rate: int = 48000,
           transition_bins: int = 1024):

    os.makedirs(output_dir, exist_ok=True)

    sub = "enhancer_v2" if use_v2 else "enhancer"
    enhancer_dir = os.path.join(model_dir, sub)
    print(f"[BWE] Loading from {enhancer_dir}")
    bwe = LavaBWE(enhancer_dir, device="cpu")

    istft_params = _get_istft_params(bwe)
    onnx_path = os.path.join(output_dir, "bwe.onnx")

    # 1-second dummy @48kHz
    dummy = torch.randn(1, sample_rate)

    # ---- Try Mode A -------------------------------------------------------
    mode = "full"
    try:
        model_a = BWEFull(bwe, cutoff_hz, sample_rate, transition_bins).eval()
        torch.onnx.export(
            model_a, dummy, onnx_path,
            input_names=["wav_48k"],
            output_names=["enhanced_48k"],
            dynamic_axes={"wav_48k": {1: "T"}, "enhanced_48k": {1: "T"}},
            opset_version=17,
            do_constant_folding=True,
            verbose=False,
        )
        print(f"[BWE] Mode A (full) export OK → {onnx_path}")
    except Exception as exc:
        print(f"[BWE] Mode A failed ({exc}), trying Mode B (split)…")
        mode = "split"
        model_b = BWESplit(bwe).eval()
        mode_b_ok = False
        last_exc = None
        # v2 MelSpectrogram needs dynamo export; v1 may need legacy trace (dynamo=False).
        for use_dynamo in (True, False):
            export_kwargs = dict(
                input_names=["wav_48k"],
                output_names=["raw_head_out"],
                dynamic_axes={"wav_48k": {1: "T"}, "raw_head_out": {2: "T_frames"}},
                opset_version=17,
                do_constant_folding=True,
                verbose=False,
            )
            if not use_dynamo:
                export_kwargs["dynamo"] = False
            try:
                torch.onnx.export(model_b, (dummy,), onnx_path, **export_kwargs)
                print(f"[BWE] Mode B (split, dynamo={use_dynamo}) export OK → {onnx_path}")
                mode_b_ok = True
                break
            except TypeError as exc:
                # dynamo= kwarg not available in older PyTorch – retry without it
                last_exc = exc
                try:
                    export_kwargs.pop("dynamo", None)
                    torch.onnx.export(model_b, (dummy,), onnx_path, **export_kwargs)
                    print(f"[BWE] Mode B (split, legacy API) export OK → {onnx_path}")
                    mode_b_ok = True
                    break
                except Exception as exc2:
                    last_exc = exc2
            except Exception as exc:
                print(f"[BWE] Mode B dynamo={use_dynamo} failed: {exc}")
                last_exc = exc
        if not mode_b_ok:
            raise RuntimeError(f"Mode B export failed: {last_exc}") from last_exc

    # ---- Write config -----------------------------------------------------
    config = {
        "mode":            mode,
        "cutoff_hz":       cutoff_hz,
        "sample_rate":     sample_rate,
        "transition_bins": transition_bins,
        **istft_params,
    }
    cfg_path = os.path.join(output_dir, "bwe_config.json")
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[BWE] Config saved → {cfg_path}")
    print(json.dumps(config, indent=2))

    # ---- Quick validation -------------------------------------------------
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        inp = dummy.numpy()
        out = sess.run(None, {sess.get_inputs()[0].name: inp})
        print(f"[BWE] ORT validation OK, output shape: {out[0].shape}")
    except ImportError:
        print("[BWE] onnxruntime not installed – skipping ORT validation")
    except Exception as e:
        print(f"[BWE] ORT validation warning: {e}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Export LavaBWE to ONNX")
    p.add_argument("model_dir",  help="HuggingFace snapshot root directory")
    p.add_argument("output_dir", help="Output directory for bwe.onnx + config")
    p.add_argument("--cutoff",       type=int,  default=7500)
    p.add_argument("--v1",           action="store_true", help="Use enhancer/ (v1) instead of enhancer_v2/")
    p.add_argument("--sample-rate",  type=int,  default=48000)
    p.add_argument("--transition-bins", type=int, default=1024)
    args = p.parse_args()

    export(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        cutoff_hz=args.cutoff,
        use_v2=not args.v1,
        sample_rate=args.sample_rate,
        transition_bins=args.transition_bins,
    )
