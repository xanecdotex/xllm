#include "mm_input_helper.h"

#include <libavformat/avformat.h>

#include <opencv2/opencv.hpp>

#include "butil/base64.h"

namespace xllm {

int FRAME_FACTOR = 2;
double FPS_DEFAULT = 2.0;
int FPS_MIN_FRAMES = 4;
int FPS_MAX_FRAMES = 768;

static inline int floor_by_factor(double x, int k) {
  return static_cast<int>(std::floor(x / k)) * k;
}

static inline int ceil_by_factor(double x, int k) {
  return static_cast<int>(std::ceil(x / k)) * k;
}

double get_avg_fps(cv::VideoCapture& cap, int64_t total_frames) {
  // 取首帧时间戳
  cap.set(cv::CAP_PROP_POS_FRAMES, 0);
  cv::Mat tmp;
  cap.read(tmp);
  double start_ms = cap.get(cv::CAP_PROP_POS_MSEC);  // 常见是 0，也可能不是

  // 取最后一帧“真正读到那一帧后的”时间戳
  int64_t last = total_frames - 1;
  cap.set(cv::CAP_PROP_POS_FRAMES, (double)last);
  cap.read(tmp);  // 必须 read 一下，很多实现只有 read 后 POS_MSEC 才更新
  double end_ms = cap.get(cv::CAP_PROP_POS_MSEC);

  // 复位
  cap.set(cv::CAP_PROP_POS_FRAMES, 0);

  // 时长 = 最后一帧时间戳 - 第一帧时间戳
  double duration_s = std::max(1e-6, (end_ms - start_ms) / 1000.0);
  LOG(INFO) << start_ms << "," << end_ms << "," << duration_s << ","
            << (double)(total_frames - 1) / duration_s;
  // 间隔数 = total_frames - 1（不是 total_frames）
  return (double)(total_frames - 1) / duration_s;
}

static int smart_nframes(int total_frames, double video_fps) {
  int minf = ceil_by_factor(FPS_MIN_FRAMES, FRAME_FACTOR);
  int maxf =
      floor_by_factor(std::min(FPS_MAX_FRAMES, total_frames), FRAME_FACTOR);
  double vf =
      (video_fps > 0.0 && !std::isnan(video_fps)) ? video_fps : FPS_DEFAULT;
  double nf_d = (double)total_frames / std::max(vf, 1e-6) * FPS_DEFAULT;
  double clamped = std::min(std::max(nf_d, (double)minf), (double)maxf);
  int nf = floor_by_factor((int)std::floor(clamped), FRAME_FACTOR);

  if (nf > total_frames) {
    LOG(FATAL) << "smart_nframes: nframes[" << nf << "] > total_frames["
               << total_frames << "]";
  }

  return nf;
}

class OpenCVImageDecoder {
 public:
  bool decode(const std::string& raw_data, torch::Tensor& t) {
    cv::Mat buffer(1, raw_data.size(), CV_8UC1, (void*)raw_data.data());
    cv::Mat image = cv::imdecode(buffer, cv::IMREAD_COLOR);
    if (image.empty()) {
      LOG(INFO) << " opencv image decode failed";
      return false;
    }

    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);  // RGB

    torch::Tensor tensor = torch::from_blob(
        image.data, {image.rows, image.cols, 3}, torch::kUInt8);

    t = tensor.permute({2, 0, 1}).clone();  // [C, H, W]
    return true;
  }
};

class OpenCVVideoDecoder {
 public:
  bool decode_from_bytes(const std::string& raw_data,
                         std::vector<torch::Tensor>& frames,
                         float& fps_out) {
    std::string tmp_path;
    if (!write_to_temp_file(raw_data, tmp_path)) {
      LOG(ERROR) << "write_to_temp_file failed";
      return false;
    }

    bool ok = decode_from_file(tmp_path, frames, fps_out);

    std::remove(tmp_path.c_str());
    return ok;
  }

  bool decode_from_file(const std::string& path,
                        std::vector<torch::Tensor>& frames,
                        float& fps_out) {
    cv::VideoCapture cap;
    if (!cap.open(path)) {
      LOG(ERROR) << "OpenCVVideoDecoder open failed: " << path;
      return false;
    }

    frames.clear();
    cv::Mat bgr;
    while (true) {
      if (!cap.read(bgr)) break;
      if (bgr.empty()) break;

      cv::Mat rgb;
      cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

      torch::Tensor t =
          torch::from_blob(rgb.data, {rgb.rows, rgb.cols, 3}, torch::kUInt8)
              .permute({2, 0, 1})  // [C, H, W]
              .clone();

      frames.emplace_back(std::move(t));
    }

    if (frames.empty()) {
      LOG(ERROR) << "OpenCVVideoDecoder got 0 frame: " << path;
      return false;
    }

    int total_frames = (int)frames.size();

    double vf = get_avg_fps(cap, total_frames);
    if (!(vf > 0.0) || std::isnan(vf)) vf = FPS_DEFAULT;

    LOG(INFO) << "total_frames=" << total_frames
              << ", video_fps=" << (double)vf;

    int nframes_len = smart_nframes(total_frames, vf);
    auto idx =
        torch::linspace(
            0, total_frames - 1, nframes_len, torch::dtype(torch::kFloat32))
            .round()
            .to(torch::kLong)
            .to(torch::kCPU)
            .contiguous();

    const int64_t* ip = idx.data_ptr<int64_t>();
    std::vector<torch::Tensor> picked;
    picked.reserve(nframes_len);
    for (int j = 0; j < nframes_len; ++j) {
      size_t k = static_cast<size_t>(ip[j]);
      picked.emplace_back(std::move(frames[k]));
    }
    frames.swap(picked);

    double sample_fps = static_cast<double>(nframes_len) /
                        std::max(1e-6, static_cast<double>(total_frames)) * vf;
    LOG(INFO) << "nframes:" << nframes_len << ", video.shape: ["
              << frames.size() << ", "
              << (frames.empty() ? 0 : frames[0].size(0)) << ", "
              << (frames.empty() ? 0 : frames[0].size(1)) << ", "
              << (frames.empty() ? 0 : frames[0].size(2)) << "]"
              << ", sample_fps=" << sample_fps;

    fps_out = (float)sample_fps;

    return true;
  }

 private:
  bool write_to_temp_file(const std::string& data, std::string& out_path) {
    char tmpl[] = "/export/home/wangziyue.28/test/temp/xllm_video_XXXXXX.mp4";
    int fd = mkstemps(tmpl, 4 /*“.mp4”长度*/);
    if (fd == -1) return false;

    FILE* f = fdopen(fd, "wb");
    if (!f) {
      close(fd);
      return false;
    }
    size_t written = fwrite(data.data(), 1, data.size(), f);
    fflush(f);
    fclose(f);

    if (written != data.size()) return false;
    out_path.assign(tmpl);
    return true;
  }
};

class Handler {
 public:
  bool process(const proto::MMInputData& msg, MMInputItem& input) {
    if (!this->load(msg, input)) {
      LOG(ERROR) << " load mm data failed";
      return false;
    }

    if (!this->decode(input)) {
      LOG(ERROR) << " decode mm data failed";
      return false;
    }

    return true;
  }

  virtual bool load(const proto::MMInputData& msg, MMInputItem& input) = 0;
  virtual bool decode(MMInputItem& input) = 0;

 protected:
  bool load_from_dataurl(const std::string& url, std::string& data) {
    size_t pos = url.find_first_of(',');
    if (pos == std::string::npos) return false;

    butil::StringPiece sub(url, pos + 1);
    return butil::Base64Decode(sub, &data);
  }

  bool load_from_local(const std::string& url, std::string& data) {
    return false;
  }

  bool load_from_http(const std::string& url, std::string& data) {
    return false;
  }
};

class ImageHandler : public Handler {
 public:
  ImageHandler() : dataurl_prefix_("data:image") {}

  virtual bool load(const proto::MMInputData& msg, MMInputItem& input) {
    input.clear();

    const auto& image_url = msg.image_url();
    const auto& url = image_url.url();

    if (url.compare(0, dataurl_prefix_.size(), dataurl_prefix_) ==
        0) {  // data url

      input.type_ = MMType::IMAGE;
      return this->load_from_dataurl(url, input.raw_data_);
    } else {
      return false;
    }
  }

  virtual bool decode(MMInputItem& input) {
    OpenCVImageDecoder decoder;
    return decoder.decode(input.raw_data_, input.decode_data_);
  }

 private:
  const std::string dataurl_prefix_;
};

class VideoHandler : public Handler {
 public:
  VideoHandler()
      : dataurl_prefix_("data:video"),
        http_prefix1_("http://"),
        http_prefix2_("https://") {}

  bool load(const proto::MMInputData& msg, MMInputItem& input) override {
    input.clear();

    const auto& video_url = msg.video_url();
    const auto& url = video_url.url();

    input.type_ = MMType::VIDEO;

    // base64
    if (url.compare(0, dataurl_prefix_.size(), dataurl_prefix_) == 0) {
      return this->load_from_dataurl(url, input.raw_data_);
    }
    // httpurl
    else if (url.compare(0, http_prefix1_.size(), http_prefix1_) == 0 ||
             url.compare(0, http_prefix2_.size(), http_prefix2_) == 0) {
      return this->load_from_http(url, input.path_);
    }
    // local
    else if (!url.empty()) {
      return this->load_from_local(url, input.path_);
    } else {
      return false;
    }
  }

  bool decode(MMInputItem& input) override {
    OpenCVVideoDecoder decoder;

    std::vector<torch::Tensor> frames;
    float fps = 2.0f;

    bool ok = false;

    // raw_data(httpurl,base64)
    if (!input.raw_data_.empty()) {
      ok = decoder.decode_from_bytes(input.raw_data_, frames, fps);
    }

    // load_from_local/http->input.path_
    else if (!input.path_.empty()) {
      ok = decoder.decode_from_file(input.path_, frames, fps);
    }

    if (!ok) {
      LOG(ERROR) << "VideoHandler decode failed";
      return false;
    }

    input.decode_data_ = torch::stack(frames, 0);  // [T,C,H,W]
    input.fps_ = fps;

    return true;
  }

 private:
  const std::string dataurl_prefix_;
  const std::string http_prefix1_, http_prefix2_;
};

class MMHandlerSet {
 public:
  MMHandlerSet() {
    handlers_["image_url"] = std::make_unique<ImageHandler>();
    handlers_["video_url"] = std::make_unique<VideoHandler>();
    // handlers_["audio_url"] = std::make_unique<AudioHandler>();
  }

  bool process(const std::string& type,
               const proto::MMInputData& msg,
               MMInputItem& input) {
    auto itor = handlers_.find(type);
    if (itor == handlers_.end()) {
      return false;
    }

    auto& handler = itor->second;
    return handler->process(msg, input);
  }

 private:
  std::unordered_map<std::string, std::unique_ptr<Handler> > handlers_;
};

MMInputHelper::MMInputHelper() {
  mm_handlers_ = std::make_unique<MMHandlerSet>();
}

MMInputHelper::~MMInputHelper() {}

bool MMInputHelper::trans(const MMChatMessageVec& vec,
                          std::vector<Message>& messages,
                          MMInputItemVec& inputs) {
  messages.clear();
  inputs.clear();

  messages.reserve(vec.size());
  inputs.reserve(vec.size());

  for (int idx = 0; idx < vec.size(); ++idx) {
    const auto& chat = vec[idx];
    const auto& role = chat.role();
    const auto& content = chat.content();

    Message::MMContentVec mmc;
    MMInputItemVec ins;
    if (!this->trans(content, mmc, ins)) {
      return false;
    }

    messages.emplace_back(role, mmc);
    inputs.insert(inputs.end(), ins.begin(), ins.end());
  }
  return true;
}

bool MMInputHelper::trans(const MMInputDataVec& vec,
                          Message::MMContentVec& mmc,
                          MMInputItemVec& inputs) {
  mmc.clear();
  inputs.clear();

  for (int idx = 0; idx < vec.size(); ++idx) {
    const auto& item = vec[idx];
    const auto& type = item.type();

    if (type == "text") {
      mmc.emplace_back(type, item.text());
    } else {
      MMInputItem input;
      if (!mm_handlers_->process(type, item, input)) {
        return false;
      }

      mmc.emplace_back(type);
      inputs.emplace_back(input);
    }
  }

  return true;
}

}  // namespace xllm
