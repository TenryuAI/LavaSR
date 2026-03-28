#pragma once
// ---------------------------------------------------------------------------
// Resampler.h – Self-contained polyphase / sinc resampler.
//
// Two primary paths are optimised:
//   • Downsampling:  arbitrary_sr → 16000 Hz  (for ULUNAS denoiser input)
//   • Upsampling  :  16000 Hz    → 48000 Hz  (3× integer ratio, polyphase FIR)
//
// For arbitrary ratios a general windowed-sinc resampler is used.
// ---------------------------------------------------------------------------
#include <vector>

namespace lavasr {

/// Resample mono float PCM from in_sr to out_sr.
/// Returns number of output samples written into 'output'.
int resample(const float* input, int n_input, int in_sr,
             int out_sr, std::vector<float>& output);

/// Resample (C, T) interleaved multi-channel audio.
/// Processes each channel independently and re-interleaves.
int resample_channels(const float* input, int n_frames, int n_channels,
                      int in_sr, int out_sr,
                      std::vector<float>& output);

} // namespace lavasr
