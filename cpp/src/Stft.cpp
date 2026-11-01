// ---------------------------------------------------------------------------
// Stft.cpp – Radix-2 FFT + STFT/ISTFT + Vocos head + FastLRMerge
//
// Performance notes (streaming BWE calls the head+ISTFT and the merge once per
// hop, so these are on the real-time path):
//   • The FFT works on split re/im float arrays with per-stage contiguous
//     twiddles, so the butterfly loops auto-vectorize. The previous kernel
//     converted every butterfly to double complex and back.
//   • Real signals are transformed with half-length complex FFTs:
//       - the merge packs orig + i*pred into ONE complex FFT and inverts with
//         an N/2 complex FFT (1.5 N-point FFTs instead of 3);
//       - the ISTFT inverts each Hermitian frame with an n_fft/2 complex FFT.
//   • The merge mask only depends on (N, sample_rate, cutoff, transition) and
//     is cached.
//
// Convention (unchanged): "forward" uses the kernel exp(+i*2*pi*k*n/N) without
// scaling, "inverse" uses exp(-i*2*pi*k*n/N) with a 1/N factor.
// ---------------------------------------------------------------------------
#include "lavasr/Stft.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <list>
#include <memory>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#if defined(_MSC_VER)
#define LAVASR_RESTRICT __restrict
#else
#define LAVASR_RESTRICT __restrict__
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

// The window, FFT twiddles and merge mask depend only on the transform size
// (and merge parameters), but were rebuilt on every stft/istft/merge call — at
// audio hop rate that is thousands of pow/cos/sin evaluations per second of
// audio. Cache per size.
//
// thread_local rather than a shared table + mutex: hosts run one LavaSR
// instance per audio channel on separate threads, and these tables are small
// (a few hundred KB at n_fft=2048). std::list keeps returned references valid
// when further sizes are added.

static const std::vector<float>& hann_for(int n) {
    struct Entry { int n; std::vector<float> win; };
    static thread_local std::list<Entry> cache;
    for (auto& e : cache) {
        if (e.n == n) return e.win;
    }
    cache.emplace_back();
    Entry& e = cache.back();
    e.n = n;
    e.win.resize(n);
    for (int i = 0; i < n; ++i) {
        e.win[i] = 0.5f * (1.0f - std::cos(2.0f * (float)M_PI * i / (n - 1)));
    }
    return e.win;
}

struct FftPlan {
    int n = 0;
    std::vector<uint32_t> rev;  // bit-reversal permutation
    // Per-stage contiguous twiddles exp(+i*2*pi*j/len) for len = 2*half:
    // the stage with `half` butterflies per block starts at offset half-1.
    std::vector<float> wr, wi;
};

static const FftPlan& plan_for(int n) {
    static thread_local std::list<FftPlan> cache;
    for (auto& e : cache) {
        if (e.n == n) return e;
    }
    cache.emplace_back();
    FftPlan& p = cache.back();
    p.n = n;
    p.rev.resize(n);
    int bits = 0;
    while ((1 << bits) < n) ++bits;
    for (int i = 0; i < n; ++i) {
        uint32_t r = 0;
        for (int b = 0; b < bits; ++b) {
            if (i & (1 << b)) r |= 1u << (bits - 1 - b);
        }
        p.rev[i] = r;
    }
    const int total = n > 1 ? n - 1 : 1;
    p.wr.assign(total, 0.0f);
    p.wi.assign(total, 0.0f);
    for (int half = 1; half < n; half <<= 1) {
        const int len = half * 2;
        for (int j = 0; j < half; ++j) {
            const double ang = 2.0 * M_PI * j / len;
            p.wr[half - 1 + j] = (float)std::cos(ang);
            p.wi[half - 1 + j] = (float)std::sin(ang);
        }
    }
    return p;
}

// ===========================================================================
// FFT kernel on split re/im arrays (no scaling).
// plus_kernel: kernel exp(+i*2*pi*k*n/N); otherwise exp(-i...).
// ===========================================================================

static void fft_soa(float* LAVASR_RESTRICT re, float* LAVASR_RESTRICT im, int n, bool plus_kernel) {
    if (n < 2) return;
    const FftPlan& p = plan_for(n);

    for (int i = 0; i < n; ++i) {
        const int j = (int)p.rev[i];
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }

    const float s = plus_kernel ? 1.0f : -1.0f;

    // len = 2: twiddle 1.
    for (int i = 0; i < n; i += 2) {
        const float ur = re[i], ui = im[i], vr = re[i + 1], vi = im[i + 1];
        re[i] = ur + vr;
        im[i] = ui + vi;
        re[i + 1] = ur - vr;
        im[i + 1] = ui - vi;
    }
    if (n < 4) return;

    // len = 4: twiddles 1 and s*i.
    for (int i = 0; i < n; i += 4) {
        {
            const float ur = re[i], ui = im[i], vr = re[i + 2], vi = im[i + 2];
            re[i] = ur + vr;
            im[i] = ui + vi;
            re[i + 2] = ur - vr;
            im[i + 2] = ui - vi;
        }
        {
            const float ur = re[i + 1], ui = im[i + 1];
            const float xr = re[i + 3], xi = im[i + 3];
            const float vr = -s * xi;  // (xr + i*xi) * (i*s)
            const float vi = s * xr;
            re[i + 1] = ur + vr;
            im[i + 1] = ui + vi;
            re[i + 3] = ur - vr;
            im[i + 3] = ui - vi;
        }
    }

    for (int len = 8; len <= n; len <<= 1) {
        const int half = len >> 1;
        const float* LAVASR_RESTRICT twr = p.wr.data() + (half - 1);
        const float* LAVASR_RESTRICT twi = p.wi.data() + (half - 1);
        for (int i = 0; i < n; i += len) {
            float* LAVASR_RESTRICT ar = re + i;
            float* LAVASR_RESTRICT ai = im + i;
            float* LAVASR_RESTRICT br = re + i + half;
            float* LAVASR_RESTRICT bi = im + i + half;
            for (int j = 0; j < half; ++j) {
                const float wr = twr[j];
                const float wi = s * twi[j];
                const float xr = br[j], xi = bi[j];
                const float vr = xr * wr - xi * wi;
                const float vi = xr * wi + xi * wr;
                const float ur = ar[j], ui = ai[j];
                ar[j] = ur + vr;
                ai[j] = ui + vi;
                br[j] = ur - vr;
                bi[j] = ui - vi;
            }
        }
    }
}

// ===========================================================================
// fft_inplace – Cooley-Tukey, radix-2, in-place (public, interleaved complex)
// ===========================================================================

void fft_inplace(std::complex<float>* data, int n, bool inverse) {
    static thread_local std::vector<float> re, im;
    if ((int)re.size() < n) {
        re.resize(n);
        im.resize(n);
    }
    for (int i = 0; i < n; ++i) {
        re[i] = data[i].real();
        im[i] = data[i].imag();
    }
    fft_soa(re.data(), im.data(), n, /*plus_kernel=*/!inverse);
    if (inverse) {
        const float inv_n = 1.0f / n;
        for (int i = 0; i < n; ++i) data[i] = {re[i] * inv_n, im[i] * inv_n};
    } else {
        for (int i = 0; i < n; ++i) data[i] = {re[i], im[i]};
    }
}

// ===========================================================================
// Real-signal inverse via a half-length complex FFT
//
// y[n] = (1/N) * sum_k M[k] * exp(-i*2*pi*k*n/N), M Hermitian, given as bins
// 0..H (H = N/2). With c[j] = y[2j] + i*y[2j+1] and C = F_H(c):
//   E[k]  = (M[k] + conj(M[H-k])) / 2
//   Od[k] = exp(-i*2*pi*k/N) * (M[k] - conj(M[H-k])) / 2
//   C[k]  = E[k] + i*Od[k]            (k = 0..H-1)
// and c = inverse_H(C). The imaginary parts of bins 0 and H do not contribute
// to a real signal; callers must pass them as zero.
// ===========================================================================

struct RealIfftRot {
    int n = 0;
    std::vector<float> c, s;  // exp(-i*2*pi*k/N) = c[k] + i*s[k], k = 0..N/2-1
};

static const RealIfftRot& rot_for(int n) {
    static thread_local std::list<RealIfftRot> cache;
    for (auto& e : cache) {
        if (e.n == n) return e;
    }
    cache.emplace_back();
    RealIfftRot& r = cache.back();
    r.n = n;
    const int h = n / 2;
    r.c.resize(h);
    r.s.resize(h);
    for (int k = 0; k < h; ++k) {
        const double ang = 2.0 * M_PI * k / n;
        r.c[k] = (float)std::cos(ang);
        r.s[k] = (float)(-std::sin(ang));
    }
    return r;
}

/// Inverse of a Hermitian spectrum given as bins 0..H (mr/mi, H+1 entries).
/// Writes N real samples to y. cr/ci are scratch of H entries each.
static void real_ifft(const float* LAVASR_RESTRICT mr, const float* LAVASR_RESTRICT mi, int N,
                      float* LAVASR_RESTRICT cr, float* LAVASR_RESTRICT ci, float* LAVASR_RESTRICT y) {
    const int H = N / 2;
    const RealIfftRot& rot = rot_for(N);
    for (int k = 0; k < H; ++k) {
        const float ar = mr[k], ai = mi[k];
        const float br = mr[H - k], bi = -mi[H - k];  // conj(M[H-k])
        const float er = 0.5f * (ar + br), ei = 0.5f * (ai + bi);
        const float tr = 0.5f * (ar - br), ti = 0.5f * (ai - bi);
        const float wc = rot.c[k], ws = rot.s[k];
        const float odr = tr * wc - ti * ws;
        const float odi = tr * ws + ti * wc;
        cr[k] = er - odi;  // C = E + i*Od
        ci[k] = ei + odr;
    }
    fft_soa(cr, ci, H, /*plus_kernel=*/false);
    const float sc = 1.0f / H;
    for (int j = 0; j < H; ++j) {
        y[2 * j] = cr[j] * sc;
        y[2 * j + 1] = ci[j] * sc;
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

    const std::vector<float>& win = hann_for(p.win_len);

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

/// Overlap-add of Hermitian frames. fill(f, mr, mi) writes bins 0..H of frame f
/// (H+1 entries each). Requires n_fft >= 4 (power of two).
template <typename FillFn>
static void istft_real_frames(FillFn&& fill, int n_frames, int n_samples, const StftParams& p,
                              std::vector<float>& out)
{
    const std::vector<float>& win = hann_for(p.win_len);
    const int N = p.n_fft;
    const int H = N / 2;

    // Window envelope normalization denominator
    const int pad        = N / 2;
    const int out_padded = (n_frames - 1) * p.hop_len + N;

    static thread_local std::vector<float> signal, env, mr, mi, cr, ci, frame;
    signal.assign(out_padded, 0.0f);
    env.assign(out_padded, 0.0f);
    mr.resize(H + 1);
    mi.resize(H + 1);
    cr.resize(H);
    ci.resize(H);
    frame.resize(N);

    for (int f = 0; f < n_frames; ++f) {
        fill(f, mr.data(), mi.data());
        mi[0] = 0.0f;  // imaginary DC / Nyquist do not contribute to a real signal
        mi[H] = 0.0f;
        real_ifft(mr.data(), mi.data(), N, cr.data(), ci.data(), frame.data());

        const int start = f * p.hop_len;
        for (int k = 0; k < N; ++k) {
            const float w = (k < p.win_len) ? win[k] : 0.0f;
            signal[start + k] += frame[k] * w;
            env[start + k]    += w * w;
        }
    }

    // Trim center-padding and normalize
    out.resize(n_samples, 0.0f);
    for (int i = 0; i < n_samples; ++i) {
        const int pi = i + pad;
        const float e = (pi < out_padded) ? env[pi] : 0.0f;
        const float s = (pi < out_padded) ? signal[pi] : 0.0f;
        out[i] = (e > 1e-8f) ? s / e : 0.0f;
    }
}

void istft(const float* spec_real, const float* spec_imag,
           int n_frames, int n_bins, int n_samples,
           const StftParams& p,
           std::vector<float>& out)
{
    const int N = p.n_fft;
    const int H = N / 2;
    if (n_bins == H + 1 && N >= 4) {
        istft_real_frames(
            [&](int f, float* mr, float* mi) {
                const size_t base = (size_t)f * n_bins;
                for (int b = 0; b <= H; ++b) {
                    mr[b] = spec_real[base + b];
                    mi[b] = spec_imag[base + b];
                }
            },
            n_frames, n_samples, p, out);
        return;
    }

    // Generic path (unusual n_bins): full Hermitian spectrum + complex IFFT.
    const std::vector<float>& win = hann_for(p.win_len);
    int pad       = N / 2;
    int out_padded = (n_frames - 1) * p.hop_len + N;

    std::vector<float> signal(out_padded, 0.0f);
    std::vector<float> env(out_padded, 0.0f);
    std::vector<std::complex<float>> frame_buf(N);

    for (int f = 0; f < n_frames; ++f) {
        size_t base = (size_t)f * n_bins;
        // Build onesided spectrum, reconstruct full Hermitian spectrum
        for (int b = 0; b < N; ++b) {
            if (b < n_bins) {
                frame_buf[b] = {spec_real[base + b], spec_imag[base + b]};
            } else {
                int mirror = N - b;
                frame_buf[b] = {spec_real[base + mirror],
                                -spec_imag[base + mirror]};
            }
        }
        fft_inplace(frame_buf.data(), N, true);  // IFFT

        int start = f * p.hop_len;
        for (int k = 0; k < N; ++k) {
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

// Branch-free float approximations of exp / sincos (abs. error ~1e-7, i.e. at
// float rounding level). The head evaluates them for ~17k bins per hop, and the
// libm calls were a third of this stage. No integer conversion of untrusted
// values, so NaN / inf inputs just propagate.

/// min(exp(x), 1e3) for float x.
static inline float exp_capped(float x) {
    const bool is_nan = x != x;
    float t = x;
    t = t > 6.9077554f ? 6.9077554f : t;    // ln(1000): result is capped there anyway
    t = t < -87.0f ? -87.0f : t;
    t = is_nan ? 0.0f : t;
    const float u = t * 1.44269504f;          // x / ln 2
    const float n = (u + 12582912.0f) - 12582912.0f;  // round to nearest (1.5 * 2^23)
    const float f = u - n;                    // [-0.5, 0.5]
    const float poly = 1.0f + f * (0.69314718f + f * (0.24022651f + f * (0.05550411f + f * (0.00961813f +
                       f * (0.00133336f + f * (0.00015404f + f * 0.0000152527f))))));
    // 2^n by assembling the exponent field (n in [-126, 10]).
    union { uint32_t i; float f; } scale;
    scale.i = (uint32_t)((int32_t)n + 127) << 23;
    const float v = poly * scale.f;
    const float capped = v > 1e3f ? 1e3f : v;
    return is_nan ? x : capped;
}

/// sin and cos of x. Reduces by integer multiples of pi to r in [-pi/2, pi/2]
/// (sin x = (-1)^m sin r, cos x = (-1)^m cos r), branch-free: random phases
/// made the earlier compare-and-flip version mispredict about half the time.
/// pi is split as 3.140625 (exact in 8 bits) + 9.6765359e-4 (Cody-Waite), good
/// to ~1e-7 for |x| up to a couple of thousand; larger arguments give a bounded
/// (but inaccurate) result, which is irrelevant for model phases.
static inline void sincos_fast(float x, float& s, float& c) {
    const float mc = x * 0.31830989f + 12582912.0f;  // integer m = round(x / pi) in the low mantissa bits
    const float m = mc - 12582912.0f;
    uint32_t mbits;
    std::memcpy(&mbits, &mc, sizeof(mbits));
    const uint32_t sign = (mbits & 1u) << 31;         // (-1)^m as a sign-bit mask
    float r = x - m * 3.140625f;
    r -= m * 9.6765359e-4f;
    r = r > 1.5707964f ? 1.5707964f : r;              // keep absurd inputs bounded
    r = r < -1.5707964f ? -1.5707964f : r;
    const float r2 = r * r;
    float sn = r * (1.0f + r2 * (-1.6666667e-1f + r2 * (8.3333333e-3f + r2 * (-1.9841270e-4f +
               r2 * (2.7557319e-6f + r2 * -2.5052108e-8f)))));
    float cs = 1.0f + r2 * (-0.5f + r2 * (4.1666668e-2f + r2 * (-1.3888889e-3f + r2 * (2.4801587e-5f +
               r2 * (-2.7557319e-7f + r2 * 2.0876757e-9f)))));
    uint32_t sb, cb;
    std::memcpy(&sb, &sn, sizeof(sb));
    std::memcpy(&cb, &cs, sizeof(cb));
    sb ^= sign;
    cb ^= sign;
    std::memcpy(&sn, &sb, sizeof(sn));
    std::memcpy(&cs, &cb, sizeof(cs));
    s = sn;
    c = cs;
}

void vocos_head_and_istft(const float* raw_head_out, int T_frames,
                          const StftParams& p, int n_samples,
                          std::vector<float>& out)
{
    const int n_bins = p.n_fft / 2 + 1;
    // raw_head_out shape: (n_fft+2, T_frames) stored as [ch * T_frames + t]
    // First n_bins channels = log-magnitude, next n_bins = phase
    if (n_samples <= 0)
        n_samples = (T_frames - 1) * p.hop_len + p.win_len;

    istft_real_frames(
        [&](int f, float* mr, float* mi) {
            for (int b = 0; b < n_bins; ++b) {
                const float log_mag = raw_head_out[(size_t)b * T_frames + f];
                const float phase   = raw_head_out[(size_t)(b + n_bins) * T_frames + f];
                const float mag     = exp_capped(log_mag);
                float sn, cs;
                sincos_fast(phase, sn, cs);
                mr[b] = mag * cs;
                mi[b] = mag * sn;
            }
        },
        T_frames, n_samples, p, out);
}

// ===========================================================================
// FastLRMerge
// ===========================================================================

struct MergeMask {
    int N = 0, sample_rate = 0, cutoff_hz = 0, trans_bins = 0;
    std::vector<float> m;  // smoothstep mask for bins 0..N/2
};

static const std::vector<float>& merge_mask_for(int N, int sample_rate, int cutoff_hz, int trans_bins) {
    static thread_local std::list<MergeMask> cache;
    for (auto& e : cache) {
        if (e.N == N && e.sample_rate == sample_rate && e.cutoff_hz == cutoff_hz && e.trans_bins == trans_bins)
            return e.m;
    }
    cache.emplace_back();
    MergeMask& e = cache.back();
    e.N = N;
    e.sample_rate = sample_rate;
    e.cutoff_hz = cutoff_hz;
    e.trans_bins = trans_bins;
    const int n_bins = N / 2 + 1;
    const double cf       = (double)cutoff_hz / ((double)sample_rate / 2.0);
    const double cutoff_b = cf * n_bins;
    const double half_tw  = trans_bins / 2.0;
    e.m.resize(n_bins);
    for (int b = 0; b < n_bins; ++b) {
        double t = ((double)b - cutoff_b) / half_tw;
        t = std::max(-1.0, std::min(1.0, t));
        const double t01 = (t + 1.0) * 0.5;
        e.m[b] = (float)(3.0 * t01 * t01 - 2.0 * t01 * t01 * t01);
    }
    return e.m;
}

void fast_lr_merge(const float* pred, const float* orig, int n,
                   int sample_rate, int cutoff_hz, int trans_bins,
                   std::vector<float>& out)
{
    const int N = next_pow2(n);  // zero-pad to power-of-2 for efficiency
    const int H = N / 2;

    static thread_local std::vector<float> zr, zi, mr, mi, cr, ci, y;
    zr.assign(N, 0.0f);
    zi.assign(N, 0.0f);
    // One complex FFT for both real signals: z = orig + i*pred.
    for (int i = 0; i < n; ++i) {
        zr[i] = orig[i];
        zi[i] = pred[i];
    }
    fft_soa(zr.data(), zi.data(), N, /*plus_kernel=*/true);

    const std::vector<float>& mask = merge_mask_for(N, sample_rate, cutoff_hz, trans_bins);

    // Separate the two spectra, blend: merged = orig + (pred - orig) * mask.
    //   O[k] = (Z[k] + conj(Z[N-k])) / 2,  P[k] = (Z[k] - conj(Z[N-k])) / (2i)
    mr.resize(H + 1);
    mi.resize(H + 1);
    for (int k = 0; k <= H; ++k) {
        const int j = (N - k) & (N - 1);
        const float zkr = zr[k], zki = zi[k];
        const float zjr = zr[j], zji = zi[j];
        const float o_r = 0.5f * (zkr + zjr), o_i = 0.5f * (zki - zji);
        const float p_r = 0.5f * (zki + zji), p_i = -0.5f * (zkr - zjr);
        const float m = mask[k];
        mr[k] = o_r + (p_r - o_r) * m;
        mi[k] = o_i + (p_i - o_i) * m;
    }
    mi[0] = 0.0f;  // real signal: DC and Nyquist are real
    mi[H] = 0.0f;

    cr.resize(H);
    ci.resize(H);
    y.resize(N);
    real_ifft(mr.data(), mi.data(), N, cr.data(), ci.data(), y.data());

    out.assign(y.begin(), y.begin() + n);
}

} // namespace lavasr
