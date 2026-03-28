// ---------------------------------------------------------------------------
// Stft.cpp – Radix-2 FFT + STFT/ISTFT + Vocos head + FastLRMerge
// ---------------------------------------------------------------------------
#include "lavasr/Stft.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace lavasr {

// ===========================================================================
// Internal helpers
// ===========================================================================

static inline int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

static void build_hann(int n, std::vector<float>& win) {
    win.resize(n);
    for (int i = 0; i < n; ++i)
        win[i] = 0.5f * (1.0f - std::cos(2.0f * (float)M_PI * i / (n - 1)));
}

// ===========================================================================
// fft_inplace – Cooley-Tukey, radix-2, in-place
// ===========================================================================

void fft_inplace(std::complex<float>* data, int n, bool inverse) {
    // Bit-reversal permutation
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(data[i], data[j]);
    }
    // Butterfly stages
    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2.0 * M_PI / len * (inverse ? -1.0 : 1.0);
        std::complex<double> wlen(std::cos(ang), std::sin(ang));
        for (int i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (int j = 0; j < len / 2; ++j) {
                auto u = std::complex<double>(data[i + j]);
                auto v = std::complex<double>(data[i + j + len / 2]) * w;
                data[i + j]           = std::complex<float>(u + v);
                data[i + j + len / 2] = std::complex<float>(u - v);
                w *= wlen;
            }
        }
    }
    if (inverse) {
        float inv_n = 1.0f / n;
        for (int i = 0; i < n; ++i) data[i] *= inv_n;
    }
}

// ===========================================================================
// STFT
// ===========================================================================

void stft(const float* samples, int n_samples,
          const StftParams& p,
          std::vector<float>& spec_real,
          std::vector<float>& spec_imag,
          int& n_frames, int& n_bins)
{
    // n_fft must be power of 2 (enforced in CMake options & docs)
    n_bins = p.n_fft / 2 + 1;
    // Number of frames (center-padded like PyTorch: pad by n_fft//2 on each side)
    int pad        = p.n_fft / 2;
    int padded_len = n_samples + 2 * pad;
    n_frames       = (padded_len - p.n_fft) / p.hop_len + 1;

    spec_real.assign((size_t)n_frames * n_bins, 0.0f);
    spec_imag.assign((size_t)n_frames * n_bins, 0.0f);

    // Build Hann window
    std::vector<float> win;
    build_hann(p.win_len, win);

    // Padded signal buffer (zero-padded edges, matching PyTorch "reflect" default
    // is actually reflect-pad, but for BWE we use constant-zero here for simplicity)
    std::vector<float> padded(padded_len, 0.0f);
    std::copy(samples, samples + n_samples, padded.data() + pad);

    std::vector<std::complex<float>> frame_buf(p.n_fft);

    for (int f = 0; f < n_frames; ++f) {
        int start = f * p.hop_len;
        // Fill frame, apply window
        for (int k = 0; k < p.n_fft; ++k) {
            int idx = start + k;
            float s = (k < p.win_len && idx < padded_len) ? padded[idx] : 0.0f;
            float w = (k < p.win_len) ? win[k] : 0.0f;
            frame_buf[k] = {s * w, 0.0f};
        }
        fft_inplace(frame_buf.data(), p.n_fft, false);
        size_t base = (size_t)f * n_bins;
        for (int b = 0; b < n_bins; ++b) {
            spec_real[base + b] = frame_buf[b].real();
            spec_imag[base + b] = frame_buf[b].imag();
        }
    }
}

// ===========================================================================
// ISTFT (weighted overlap-add)
// ===========================================================================

void istft(const float* spec_real, const float* spec_imag,
           int n_frames, int n_bins, int n_samples,
           const StftParams& p,
           std::vector<float>& out)
{
    std::vector<float> win;
    build_hann(p.win_len, win);

    // Window envelope normalization denominator
    int pad       = p.n_fft / 2;
    int out_padded = (n_frames - 1) * p.hop_len + p.n_fft;

    std::vector<float> signal(out_padded, 0.0f);
    std::vector<float> env(out_padded, 0.0f);

    std::vector<std::complex<float>> frame_buf(p.n_fft);

    for (int f = 0; f < n_frames; ++f) {
        size_t base = (size_t)f * n_bins;
        // Build onesided spectrum, reconstruct full Hermitian spectrum
        for (int b = 0; b < p.n_fft; ++b) {
            if (b < n_bins) {
                frame_buf[b] = {spec_real[base + b], spec_imag[base + b]};
            } else {
                int mirror = p.n_fft - b;
                frame_buf[b] = {spec_real[base + mirror],
                                -spec_imag[base + mirror]};
            }
        }
        fft_inplace(frame_buf.data(), p.n_fft, true);  // IFFT

        int start = f * p.hop_len;
        for (int k = 0; k < p.n_fft; ++k) {
            float w = (k < p.win_len) ? win[k] : 0.0f;
            signal[start + k] += frame_buf[k].real() * w;
            env[start + k]    += w * w;
        }
    }

    // Trim center-padding and normalize
    out.resize(n_samples, 0.0f);
    for (int i = 0; i < n_samples; ++i) {
        int pi = i + pad;
        float e = (pi < (int)env.size()) ? env[pi] : 0.0f;
        float s = (pi < (int)signal.size()) ? signal[pi] : 0.0f;
        out[i] = (e > 1e-8f) ? s / e : 0.0f;
    }
}

// ===========================================================================
// Vocos head custom_forward + ISTFT  (split-mode BWE)
// ===========================================================================

void vocos_head_and_istft(const float* raw_head_out, int T_frames,
                          const StftParams& p, int n_samples,
                          std::vector<float>& out)
{
    int n_bins = p.n_fft / 2 + 1;
    // raw_head_out shape: (n_fft+2, T_frames) stored as [ch * T_frames + t]
    // First n_bins channels = log-magnitude, next n_bins = phase
    std::vector<float> spec_real((size_t)T_frames * n_bins);
    std::vector<float> spec_imag((size_t)T_frames * n_bins);

    for (int f = 0; f < T_frames; ++f) {
        for (int b = 0; b < n_bins; ++b) {
            float log_mag = raw_head_out[(size_t)b * T_frames + f];
            float phase   = raw_head_out[(size_t)(b + n_bins) * T_frames + f];
            float mag     = std::min(std::exp(log_mag), 1e3f);
            spec_real[(size_t)f * n_bins + b] = mag * std::cos(phase);
            spec_imag[(size_t)f * n_bins + b] = mag * std::sin(phase);
        }
    }

    if (n_samples <= 0)
        n_samples = (T_frames - 1) * p.hop_len + p.win_len;

    istft(spec_real.data(), spec_imag.data(),
          T_frames, n_bins, n_samples, p, out);
}

// ===========================================================================
// FastLRMerge
// ===========================================================================

void fast_lr_merge(const float* pred, const float* orig, int n,
                   int sample_rate, int cutoff_hz, int trans_bins,
                   std::vector<float>& out)
{
    int N = next_pow2(n);  // zero-pad to power-of-2 for efficiency

    std::vector<std::complex<float>> buf_p(N, {0, 0});
    std::vector<std::complex<float>> buf_o(N, {0, 0});
    for (int i = 0; i < n; ++i) {
        buf_p[i] = {pred[i], 0.0f};
        buf_o[i] = {orig[i], 0.0f};
    }
    fft_inplace(buf_p.data(), N, false);
    fft_inplace(buf_o.data(), N, false);

    // Smoothstep mask (onesided bins 0..N/2)
    int n_bins       = N / 2 + 1;
    double cf        = (double)cutoff_hz / ((double)sample_rate / 2.0);
    double cutoff_b  = cf * n_bins;
    double half_tw   = trans_bins / 2.0;

    for (int b = 0; b < N; ++b) {
        // Mirror: work on the full spectrum symmetrically
        int eb = (b <= N / 2) ? b : (N - b);
        double t   = ((double)eb - cutoff_b) / half_tw;
        t = std::max(-1.0, std::min(1.0, t));
        double t01 = (t + 1.0) * 0.5;
        float mask  = (float)(3.0 * t01 * t01 - 2.0 * t01 * t01 * t01);
        // merged = orig + (pred - orig) * mask
        buf_o[b] += (buf_p[b] - buf_o[b]) * mask;
    }

    fft_inplace(buf_o.data(), N, true);  // IFFT

    out.resize(n);
    for (int i = 0; i < n; ++i) out[i] = buf_o[i].real();
}

} // namespace lavasr
