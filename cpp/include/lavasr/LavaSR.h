#pragma once
// ---------------------------------------------------------------------------
// LavaSR.h – Public C++ and C API for the LavaSR audio enhancement engine.
//
// C++ usage:
//   lavasr::LavaSRConfig cfg; cfg.cutoff_hz = 7500;
//   lavasr::LavaSR enhancer("bwe.onnx", "denoiser.onnx", cfg);
//   auto out = enhancer.enhance(pcm.data(), n_frames, 16000, 2);
//   // out: interleaved float32 @ 48 kHz
//
// C / FFI usage (Swift, Kotlin JNI, Python ctypes):
//   LavaSRHandle h = lavasr_create("bwe.onnx", "denoiser.onnx", 7500, 0);
//   float* out; int out_n;
//   lavasr_enhance(h, pcm, n, 16000, 2, 0, &out, &out_n);
//   lavasr_free_buffer(out);
//   lavasr_free(h);
// ---------------------------------------------------------------------------
#include <memory>
#include <string>
#include <vector>

namespace lavasr {

struct LavaSRConfig {
    int  cutoff_hz       = 7500;   ///< FastLRMerge blend frequency (Hz)
    bool denoise         = false;  ///< Run ULUNAS denoiser before BWE
    bool batch_mode      = false;  ///< Split very long audio into ~1.28 s chunks
    int  batch_chunk_ms  = 1280;
};

class LavaSRImpl;  // forward-declared; defined in LavaSR.cpp

class LavaSR {
public:
    /// @param bwe_onnx_path       Path to bwe.onnx (and bwe_config.json alongside).
    /// @param denoiser_onnx_path  Path to denoiser.onnx (and denoiser_config.json alongside).
    LavaSR(const std::string& bwe_onnx_path,
           const std::string& denoiser_onnx_path,
           const LavaSRConfig& config = {});

    ~LavaSR();

    /// Enhance interleaved float32 PCM.
    ///
    /// @param samples     Interleaved PCM input at in_sr.
    /// @param n_frames    Number of audio frames (samples / n_channels).
    /// @param in_sr       Input sample rate (any value; resampled internally to 16 kHz).
    /// @param n_channels  1 (mono) or 2 (stereo); channels processed independently.
    /// @return            Interleaved float32 PCM at 48 kHz.
    std::vector<float> enhance(const float* samples, int n_frames,
                               int in_sr, int n_channels);

private:
    std::unique_ptr<LavaSRImpl> impl_;
};

} // namespace lavasr

// ===========================================================================
// C interface  (extern "C" so it can be called from Swift, Kotlin JNI, etc.)
// ===========================================================================
#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to a LavaSR instance.
typedef void* LavaSRHandle;

/// Create a LavaSR enhancer.
/// @param bwe_onnx        Path to bwe.onnx.
/// @param denoiser_onnx   Path to denoiser.onnx.
/// @param cutoff_hz       FastLRMerge cutoff (Hz); pass 0 to use default (7500).
/// @param denoise         Non-zero to enable denoiser.
/// @return                Opaque handle, or NULL on failure.
LavaSRHandle lavasr_create(const char* bwe_onnx,
                           const char* denoiser_onnx,
                           int cutoff_hz,
                           int denoise);

/// Enhance audio.
///
/// @param handle       Handle returned by lavasr_create().
/// @param in_samples   Interleaved float32 PCM at in_sr.
/// @param in_frames    Number of frames (total samples = in_frames * n_channels).
/// @param in_sr        Input sample rate.
/// @param n_channels   1 or 2.
/// @param denoise      Override denoise flag (0 = off, 1 = on, -1 = use handle setting).
/// @param out_samples  [out] Caller receives a heap-allocated buffer; free with lavasr_free_buffer().
/// @param out_frames   [out] Number of frames in out_samples (at 48 kHz).
/// @return             0 on success, negative on error.
int lavasr_enhance(LavaSRHandle handle,
                   const float* in_samples, int in_frames,
                   int in_sr, int n_channels, int denoise,
                   float** out_samples, int* out_frames);

/// Destroy a LavaSR instance created by lavasr_create().
void lavasr_free(LavaSRHandle handle);

/// Free a buffer returned by lavasr_enhance().
void lavasr_free_buffer(float* buf);

#ifdef __cplusplus
} // extern "C"
#endif
