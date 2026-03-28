#pragma once
// ---------------------------------------------------------------------------
// Stft.h – STFT / ISTFT implementation using a self-contained radix-2 FFT.
//
// Used by:
//   • LavaDenoiser: STFT pre-processing and ISTFT post-processing for ULUNAS
//   • LavaBWE (split mode): ISTFT for Vocos head + FastLRMerge
// ---------------------------------------------------------------------------
#include <complex>
#include <vector>
#include <cstddef>

namespace lavasr {

struct StftParams {
    int n_fft    = 512;   // must be power of 2
    int hop_len  = 256;
    int win_len  = 512;
    // sample_rate used for FastLRMerge – not needed for STFT itself
};

// ---- Raw FFT ---------------------------------------------------------------

/// In-place radix-2 Cooley-Tukey FFT.  n must be a power of two.
void fft_inplace(std::complex<float>* data, int n, bool inverse);

// ---- STFT ------------------------------------------------------------------

/// Forward STFT (onesided, Hann window).
///
/// @param samples   Mono float32 PCM, length n_samples.
/// @param n_samples Number of input samples.
/// @param p         STFT parameters.
/// @param spec_real [out] Real parts,  layout [frame][bin], total n_frames * n_bins floats.
/// @param spec_imag [out] Imag parts,  same layout.
/// @param n_frames  [out] Number of frames produced.
/// @param n_bins    [out] Number of frequency bins  = n_fft/2 + 1.
void stft(const float* samples, int n_samples,
          const StftParams& p,
          std::vector<float>& spec_real,
          std::vector<float>& spec_imag,
          int& n_frames, int& n_bins);

// ---- ISTFT -----------------------------------------------------------------

/// Inverse STFT (overlap-add, Hann window).
///
/// @param spec_real  Real parts,  layout [frame][bin].
/// @param spec_imag  Imag parts,  layout [frame][bin].
/// @param n_frames   Number of frames.
/// @param n_bins     Must equal n_fft/2 + 1.
/// @param n_samples  Target output length; result is zero-padded or trimmed.
/// @param p          STFT parameters (must match the forward call).
/// @param out        [out] Reconstructed PCM samples (length n_samples).
void istft(const float* spec_real, const float* spec_imag,
           int n_frames, int n_bins, int n_samples,
           const StftParams& p,
           std::vector<float>& out);

// ---- Vocos head (split-mode BWE) ------------------------------------------

/// Apply the custom Vocos ISTFT head logic on raw_head_out.
///
/// raw_head_out layout: (n_fft+2) * T_frames  interleaved as
///     [mag_ch0..mag_chF][phase_ch0..phase_chF] for every frame
/// i.e. raw_head_out[ch * T_frames + t]  where ch ∈ [0, n_fft+2).
///
/// @param raw_head_out Pointer to (n_fft+2) * T_frames floats.
/// @param T_frames     Number of time frames.
/// @param p            STFT params (n_fft, hop_len, win_len) for the BWE model.
/// @param n_samples    Target waveform length (0 → infer from frames).
/// @param out          [out] Reconstructed waveform.
void vocos_head_and_istft(const float* raw_head_out, int T_frames,
                          const StftParams& p, int n_samples,
                          std::vector<float>& out);

// ---- FastLRMerge -----------------------------------------------------------

/// Frequency-domain blend: low-freq from orig, high-freq from pred.
/// Uses next-power-of-2 RFFT internally (results are equivalent).
///
/// @param pred         BWE-predicted waveform (length n).
/// @param orig         Original waveform (length n).
/// @param n            Signal length.
/// @param sample_rate  Sample rate (for computing cutoff bin).
/// @param cutoff_hz    Blend boundary (pred dominates above, orig below).
/// @param trans_bins   Width of the smoothstep transition band in FFT bins.
/// @param out          [out] Merged waveform (length n).
void fast_lr_merge(const float* pred, const float* orig, int n,
                   int sample_rate, int cutoff_hz, int trans_bins,
                   std::vector<float>& out);

} // namespace lavasr
