// ---------------------------------------------------------------------------
// AudioIO.cpp – WAV I/O via dr_wav (fetched by CMake).
// ---------------------------------------------------------------------------

// dr_wav is a single-header library: one translation unit owns the impl.
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include "lavasr/AudioIO.h"
#include <cstring>
#include <cstdio>

namespace lavasr {

AudioData load_wav(const std::string& path) {
    AudioData ad;
    drwav wav;

    if (!drwav_init_file(&wav, path.c_str(), nullptr)) {
        fprintf(stderr, "[LavaSR] Failed to open WAV: %s\n", path.c_str());
        return ad;
    }

    ad.sample_rate = (int)wav.sampleRate;
    ad.n_channels  = (int)wav.channels;
    ad.n_frames    = (int)wav.totalPCMFrameCount;

    ad.samples.resize((size_t)ad.n_frames * ad.n_channels);
    drwav_read_pcm_frames_f32(&wav, (drwav_uint64)ad.n_frames, ad.samples.data());
    drwav_uninit(&wav);

    return ad;
}

bool save_wav(const std::string& path,
              const float* samples,
              int n_frames,
              int n_channels,
              int sample_rate)
{
    drwav_data_format fmt;
    fmt.container     = drwav_container_riff;
    fmt.format        = DR_WAVE_FORMAT_IEEE_FLOAT;
    fmt.channels      = (drwav_uint32)n_channels;
    fmt.sampleRate    = (drwav_uint32)sample_rate;
    fmt.bitsPerSample = 32;

    drwav wav;
    if (!drwav_init_file_write(&wav, path.c_str(), &fmt, nullptr)) {
        fprintf(stderr, "[LavaSR] Failed to create WAV: %s\n", path.c_str());
        return false;
    }

    drwav_write_pcm_frames(&wav, (drwav_uint64)n_frames, samples);
    drwav_uninit(&wav);
    return true;
}

} // namespace lavasr
