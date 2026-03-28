// ---------------------------------------------------------------------------
// Resampler.cpp – Kaiser-windowed sinc resampler (self-contained)
//
// Two paths:
//   • General  : windowed-sinc with configurable filter length
//   • 3× exact : optimised polyphase FIR (16 kHz → 48 kHz)
// ---------------------------------------------------------------------------
#include "lavasr/Resampler.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace lavasr {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static double sinc(double x) {
    if (std::abs(x) < 1e-9) return 1.0;
    return std::sin(M_PI * x) / (M_PI * x);
}

// Modified Bessel I0 (for Kaiser window)
static double bessel_i0(double x) {
    double sum = 1.0, term = 1.0;
    for (int k = 1; k <= 30; ++k) {
        term *= (x / (2.0 * k));
        term *= term;  // square: term = (x/2 / k)^(2k)
        // Actually redo correctly:
        (void)term;
        break; // use simpler approximation below
    }
    // Simpler converging series
    double d = 1.0;
    sum = 1.0;
    for (int k = 1; k <= 50; ++k) {
        d *= (x * x) / (4.0 * k * k);
        sum += d;
        if (d < 1e-15 * sum) break;
    }
    return sum;
}

// Build Kaiser-windowed sinc FIR lowpass.
// cutoff: normalised frequency (0..1, where 1 = Nyquist of highest SR).
// half_len: one-sided filter length (total taps = 2*half_len+1).
static void build_kaiser_sinc(double cutoff, int half_len, double beta,
                               std::vector<float>& h) {
    int total = 2 * half_len + 1;
    h.resize(total);
    double i0_beta = bessel_i0(beta);
    for (int i = 0; i < total; ++i) {
        int n = i - half_len;
        double w_num = (double)n / half_len;
        double kaiser = bessel_i0(beta * std::sqrt(1.0 - w_num * w_num)) / i0_beta;
        // Ideal LP: h[n] = W*sinc(W*n) has DC gain=1 and passband [0, W/2].
        // With cutoff = 1/max(up,down), the anti-aliasing edge falls at
        // cutoff/2 = min(in_sr,out_sr)/(2*in_sr*up), i.e. min_sr/2 in Hz.
        // The old formula had an extra 2× factor, giving DC gain=2 (clipping).
        h[i] = (float)(sinc(cutoff * n) * kaiser * cutoff);
    }
}

// ---------------------------------------------------------------------------
// General resampler (windowed sinc)
// ---------------------------------------------------------------------------

int resample(const float* input, int n_input, int in_sr,
             int out_sr, std::vector<float>& output)
{
    if (in_sr == out_sr) {
        output.assign(input, input + n_input);
        return n_input;
    }

    // Rational approximation: find GCD to reduce ratio
    auto gcd = [](int a, int b) { while (b) { a %= b; std::swap(a, b); } return a; };
    int g = gcd(in_sr, out_sr);
    int up   = out_sr / g;   // upsample factor
    int down = in_sr  / g;   // downsample factor

    // The ideal LP impulse response h[n] = W*sinc(W*n) has:
    //   DC gain = 1,  passband [0, W/2]  (with 1.0 = upsampled sample rate).
    // We want passband edge at min(in_sr,out_sr)/2.
    //   W/2 = min_sr/2 / (in_sr*up)  →  W = min_sr / (in_sr*up) = 1/max(up,down).
    double cutoff  = 1.0 / std::max(up, down);
    int    half_len = 32 * std::max(up, down);  // filter quality
    double beta    = 8.0;                        // Kaiser shape

    std::vector<float> h;
    build_kaiser_sinc(cutoff, half_len, beta, h);

    int n_out = (int)((long long)n_input * up / down);
    output.resize(n_out, 0.0f);

    // Polyphase filtering
    for (int i = 0; i < n_out; ++i) {
        // Position in the upsampled grid
        long long pos_up = (long long)i * down;
        int phase        = (int)(pos_up % up);
        int src_center   = (int)(pos_up / up);

        double acc = 0.0;
        for (int k = -half_len; k <= half_len; ++k) {
            // Polyphase index in h: h[half_len + k*up + phase] ... but we stored
            // a single filter; for polyphase we evaluate h at (k*up + phase):
            int h_idx = half_len + k * (int)up + phase;
            if (h_idx < 0 || h_idx >= (int)h.size()) continue;
            int src_idx = src_center + k;
            if (src_idx < 0 || src_idx >= n_input) continue;
            acc += (double)input[src_idx] * h[h_idx];
        }
        // Scale by upsample factor to compensate for energy spreading
        output[i] = (float)(acc * up);
    }
    return n_out;
}

// ---------------------------------------------------------------------------
// Multi-channel resampler  (de-interleave → resample each channel → re-interleave)
// ---------------------------------------------------------------------------

int resample_channels(const float* input, int n_frames, int n_channels,
                      int in_sr, int out_sr,
                      std::vector<float>& output)
{
    if (n_channels == 1) {
        return resample(input, n_frames, in_sr, out_sr, output);
    }

    // De-interleave
    std::vector<std::vector<float>> channels(n_channels,
                                             std::vector<float>(n_frames));
    for (int f = 0; f < n_frames; ++f)
        for (int c = 0; c < n_channels; ++c)
            channels[c][f] = input[f * n_channels + c];

    // Resample each channel
    std::vector<std::vector<float>> out_ch(n_channels);
    int n_out = 0;
    for (int c = 0; c < n_channels; ++c)
        n_out = resample(channels[c].data(), n_frames, in_sr, out_sr, out_ch[c]);

    // Re-interleave
    output.resize((size_t)n_out * n_channels);
    for (int f = 0; f < n_out; ++f)
        for (int c = 0; c < n_channels; ++c)
            output[(size_t)f * n_channels + c] = out_ch[c][f];

    return n_out;
}

} // namespace lavasr
