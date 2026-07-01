#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

scratch_root="${VLA_JEPA_SCRATCH:-${repo_root}}"
src_dir="${VLA_JEPA_DECORD_SRC:-${scratch_root}/src/decord}"
wheelhouse="${VLA_JEPA_WHEELHOUSE:-${scratch_root}/wheelhouse}"
python_bin="${VLA_JEPA_ENV_PYTHON:-${repo_root}/.venv/bin/python}"
decord_ref="${VLA_JEPA_DECORD_REF:-v0.6.0}"
jobs="${VLA_JEPA_DECORD_BUILD_JOBS:-$(nproc)}"

if [[ -z "${CUDA_HOME:-}" ]]; then
  for candidate in /usr/local/cuda-12.4 /usr/local/cuda; do
    if [[ -x "${candidate}/bin/nvcc" ]]; then
      export CUDA_HOME="${candidate}"
      break
    fi
  done
fi
if [[ -z "${CUDA_HOME:-}" || ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "Missing CUDA_HOME/bin/nvcc." >&2
  exit 1
fi
export PATH="${CUDA_HOME}/bin:${PATH}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_cmd git
require_cmd cmake
require_cmd make
require_cmd pkg-config

if [[ ! -x "${python_bin}" ]]; then
  echo "Missing Python environment at ${python_bin}" >&2
  exit 1
fi

mkdir -p "$(dirname "${src_dir}")" "${wheelhouse}"

if [[ ! -d "${src_dir}/.git" ]]; then
  git clone --recursive --branch "${decord_ref}" https://github.com/dmlc/decord.git "${src_dir}"
else
  git -C "${src_dir}" fetch --tags
  git -C "${src_dir}" checkout "${decord_ref}"
  git -C "${src_dir}" submodule update --init --recursive
fi

if ! grep -q "libavcodec/bsf.h" "${src_dir}/src/video/ffmpeg/ffmpeg_common.h"; then
  git -C "${src_dir}" apply <<'PATCH'
diff --git a/src/video/ffmpeg/ffmpeg_common.h b/src/video/ffmpeg/ffmpeg_common.h
index b0b973f..f0f7316 100644
--- a/src/video/ffmpeg/ffmpeg_common.h
+++ b/src/video/ffmpeg/ffmpeg_common.h
@@ -21,6 +21,7 @@
 extern "C" {
 #endif
 #include <libavcodec/avcodec.h>
+#include <libavcodec/bsf.h>
 #include <libavformat/avformat.h>
 #include <libavformat/avio.h>
 #include <libavfilter/avfilter.h>
diff --git a/src/video/nvcodec/cuda_threaded_decoder.cc b/src/video/nvcodec/cuda_threaded_decoder.cc
index 62bc7ee..957a90d 100644
--- a/src/video/nvcodec/cuda_threaded_decoder.cc
+++ b/src/video/nvcodec/cuda_threaded_decoder.cc
@@ -17,7 +17,7 @@ namespace decord {
 namespace cuda {
 using namespace runtime;

-CUThreadedDecoder::CUThreadedDecoder(int device_id, AVCodecParameters *codecpar, AVInputFormat *iformat)
+CUThreadedDecoder::CUThreadedDecoder(int device_id, AVCodecParameters *codecpar, const AVInputFormat *iformat)
     : device_id_(device_id), stream_({device_id, false}), device_{}, ctx_{}, parser_{}, decoder_{},
     pkt_queue_{}, frame_queue_{},
     run_(false), frame_count_(0), draining_(false),
@@ -70,7 +70,7 @@ CUThreadedDecoder::CUThreadedDecoder(int device_id, AVCodecParameters *codecpar,
     }
 }

-void CUThreadedDecoder::InitBitStreamFilter(AVCodecParameters *codecpar, AVInputFormat *iformat) {
+void CUThreadedDecoder::InitBitStreamFilter(AVCodecParameters *codecpar, const AVInputFormat *iformat) {
     const char* bsf_name = nullptr;
     if (AV_CODEC_ID_H264 == codecpar->codec_id) {
         // H.264
diff --git a/src/video/nvcodec/cuda_threaded_decoder.h b/src/video/nvcodec/cuda_threaded_decoder.h
index d7e6fcd..61958a1 100644
--- a/src/video/nvcodec/cuda_threaded_decoder.h
+++ b/src/video/nvcodec/cuda_threaded_decoder.h
@@ -46,7 +46,7 @@ class CUThreadedDecoder final : public ThreadedDecoderInterface {
     using FrameOrderQueuePtr = std::unique_ptr<FrameOrderQueue>;

     public:
-        CUThreadedDecoder(int device_id, AVCodecParameters *codecpar, AVInputFormat *iformat);
+        CUThreadedDecoder(int device_id, AVCodecParameters *codecpar, const AVInputFormat *iformat);
         void SetCodecContext(AVCodecContext *dec_ctx, int width = -1, int height = -1, int rotation = 0);
         bool Initialized() const;
         void Start();
@@ -70,7 +70,7 @@ class CUThreadedDecoder final : public ThreadedDecoderInterface {
         void LaunchThreadImpl();
         void RecordInternalError(std::string message);
         void CheckErrorStatus();
-        void InitBitStreamFilter(AVCodecParameters *codecpar, AVInputFormat *iformat);
+        void InitBitStreamFilter(AVCodecParameters *codecpar, const AVInputFormat *iformat);

         int device_id_;
         CUStream stream_;
diff --git a/src/video/video_reader.cc b/src/video/video_reader.cc
index af4858d..99c9635 100644
--- a/src/video/video_reader.cc
+++ b/src/video/video_reader.cc
@@ -145,7 +145,7 @@ VideoReader::~VideoReader(){

 void VideoReader::SetVideoStream(int stream_nb) {
     if (!fmt_ctx_) return;
-    AVCodec *dec;
+    const AVCodec *dec;
     int st_nb = av_find_best_stream(fmt_ctx_.get(), AVMEDIA_TYPE_VIDEO, stream_nb, -1, &dec, 0);
     // LOG(INFO) << "find best stream: " << st_nb;
     CHECK_GE(st_nb, 0) << "ERROR cannot find video stream with wanted index: " << stream_nb;
PATCH
fi

nvml_lib="${VLA_JEPA_CUDA_NVIDIA_ML_LIBRARY:-/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1}"
if [[ ! -e "${nvml_lib}" ]]; then
  nvml_lib="/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
fi

rm -rf "${src_dir}/build"
cmake -S "${src_dir}" -B "${src_dir}/build" \
  -DUSE_CUDA="${CUDA_HOME}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="${VLA_JEPA_DECORD_CUDA_ARCHITECTURES:-80}" \
  -DCUDA_NVIDIA_ML_LIBRARY="${nvml_lib}"
cmake --build "${src_dir}/build" -j "${jobs}"

DECORD_LIBRARY_PATH="${src_dir}/build" "${python_bin}" -m pip wheel --no-deps "${src_dir}/python" -w "${wheelhouse}"
wheel="$(find "${wheelhouse}" -maxdepth 1 -name 'decord-0.6.0-*.whl' -type f | sort | tail -1)"
if [[ -z "${wheel}" ]]; then
  echo "Failed to build Decord wheel into ${wheelhouse}" >&2
  exit 1
fi
"${python_bin}" -m pip install --force-reinstall --no-deps "${wheel}"

echo "Installed CUDA-enabled Decord wheel: ${wheel}"
