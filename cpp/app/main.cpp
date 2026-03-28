// ---------------------------------------------------------------------------
// main.cpp – LavaSR command-line tool
//
// Usage:
//   lavasr [options] input.wav output.wav
//
// Options:
//   --bwe        <path>   Path to bwe.onnx    [required]
//   --denoiser   <path>   Path to denoiser.onnx [required]
//   --cutoff     <hz>     FastLRMerge cutoff frequency (default: 7500)
//   --denoise             Enable ULUNAS denoiser (default: off)
//   --batch               Enable chunked processing for long audio
//   --input-sr   <hz>     Override input sample rate (0 = read from WAV header)
//   --help
// ---------------------------------------------------------------------------

#include "lavasr/LavaSR.h"
#include "lavasr/AudioIO.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void print_usage(const char* argv0) {
    fprintf(stderr,
        "Usage: %s [options] input.wav output.wav\n\n"
        "Options:\n"
        "  --bwe       <path>   Path to bwe.onnx (required)\n"
        "  --denoiser  <path>   Path to denoiser.onnx (required)\n"
        "  --cutoff    <hz>     Blend cutoff Hz (default: 7500)\n"
        "  --denoise            Enable denoiser (default: off)\n"
        "  --batch              Chunked processing for long audio\n"
        "  --input-sr  <hz>     Override input sample rate\n"
        "  --help\n",
        argv0);
}

int main(int argc, char* argv[]) {
    if (argc < 2) { print_usage(argv[0]); return 1; }

    std::string bwe_path, den_path, input_path, output_path;
    lavasr::LavaSRConfig cfg;
    int override_sr = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help") { print_usage(argv[0]); return 0; }
        else if (arg == "--bwe"      && i+1 < argc) { bwe_path  = argv[++i]; }
        else if (arg == "--denoiser" && i+1 < argc) { den_path  = argv[++i]; }
        else if (arg == "--cutoff"   && i+1 < argc) { cfg.cutoff_hz  = std::atoi(argv[++i]); }
        else if (arg == "--denoise")                 { cfg.denoise    = true; }
        else if (arg == "--batch")                   { cfg.batch_mode = true; }
        else if (arg == "--input-sr" && i+1 < argc) { override_sr = std::atoi(argv[++i]); }
        else if (arg[0] != '-') {
            if (input_path.empty())       input_path  = arg;
            else if (output_path.empty()) output_path = arg;
        } else {
            fprintf(stderr, "Unknown option: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    if (bwe_path.empty() || den_path.empty()) {
        fprintf(stderr, "Error: --bwe and --denoiser are required.\n");
        print_usage(argv[0]);
        return 1;
    }
    if (input_path.empty() || output_path.empty()) {
        fprintf(stderr, "Error: input and output WAV paths are required.\n");
        print_usage(argv[0]);
        return 1;
    }

    // ---- Load input WAV --------------------------------------------------
    auto audio = lavasr::load_wav(input_path);
    if (audio.n_frames == 0) {
        fprintf(stderr, "Error: failed to load %s\n", input_path.c_str());
        return 1;
    }

    int in_sr = (override_sr > 0) ? override_sr : audio.sample_rate;
    fprintf(stderr,
        "[LavaSR] Input : %s  (%d Hz, %d ch, %d frames)\n",
        input_path.c_str(), in_sr, audio.n_channels, audio.n_frames);
    fprintf(stderr,
        "[LavaSR] Config: cutoff=%d Hz, denoise=%s, batch=%s\n",
        cfg.cutoff_hz, cfg.denoise ? "on" : "off", cfg.batch_mode ? "on" : "off");
    fprintf(stderr, "[LavaSR] BWE      : %s\n", bwe_path.c_str());
    fprintf(stderr, "[LavaSR] Denoiser : %s\n", den_path.c_str());

    // ---- Create enhancer ------------------------------------------------
    lavasr::LavaSR enhancer(bwe_path, den_path, cfg);

    // ---- Enhance ---------------------------------------------------------
    auto out = enhancer.enhance(audio.samples.data(),
                                audio.n_frames,
                                in_sr,
                                audio.n_channels);

    int out_frames = (int)(out.size() / audio.n_channels);
    fprintf(stderr, "[LavaSR] Output: %d frames @ 48000 Hz, %d ch\n",
            out_frames, audio.n_channels);

    // ---- Save output WAV ------------------------------------------------
    if (!lavasr::save_wav(output_path, out.data(), out_frames,
                          audio.n_channels, 48000)) {
        fprintf(stderr, "Error: failed to write %s\n", output_path.c_str());
        return 1;
    }
    fprintf(stderr, "[LavaSR] Saved → %s\n", output_path.c_str());
    return 0;
}
