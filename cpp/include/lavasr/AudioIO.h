#pragma once
// ---------------------------------------------------------------------------
// AudioIO.h – WAV file I/O using the dr_wav single-header library (MIT).
// ---------------------------------------------------------------------------
#include <string>
#include <vector>

namespace lavasr {

struct AudioData {
    std::vector<float> samples;  ///< interleaved float32 PCM, length = n_channels * n_frames
    int sample_rate  = 0;
    int n_channels   = 0;
    int n_frames     = 0;
};

/// Load a WAV file.  Returns an empty AudioData on failure (n_frames == 0).
AudioData load_wav(const std::string& path);

/// Save interleaved float32 PCM as a WAV file.
/// @return true on success.
bool save_wav(const std::string& path,
              const float* samples,   ///< interleaved, length = n_channels * n_frames
              int n_frames,
              int n_channels,
              int sample_rate);

} // namespace lavasr
