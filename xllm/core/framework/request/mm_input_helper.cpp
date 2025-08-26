#include "mm_input_helper.h"

#include <opencv2/opencv.hpp>

#include "butil/base64.h"

namespace xllm {

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
    // 1) 将 raw bytes 落盘为临时文件（cv::VideoCapture 不支持直接从内存解码）
    std::string tmp_path;
    if (!write_to_temp_file(raw_data, tmp_path)) {
      LOG(ERROR) << "write_to_temp_file failed";
      return false;
    }

    bool ok = decode_from_file(tmp_path, frames, fps_out);

    // 清理临时文件
    // std::remove(tmp_path.c_str());
    return ok;
  }

  // 从本地文件路径解码
  bool decode_from_file(const std::string& path,
                        std::vector<torch::Tensor>& frames,
                        float& fps_out) {
    cv::VideoCapture cap;
    if (!cap.open(path)) {
      LOG(ERROR) << "OpenCVVideoDecoder open failed: " << path;
      return false;
    }

    // 读取 FPS（可能为 0/NaN）
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (!(fps > 0.0) || std::isnan(fps)) fps = 2.0;  // 与你 Python 侧默认一致
    fps_out = static_cast<float>(fps);

    frames.clear();
    cv::Mat bgr;
    while (true) {
      if (!cap.read(bgr)) break;
      if (bgr.empty()) break;

      cv::Mat rgb;
      cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

      // [H, W, 3] → [C, H, W]，uint8；clone 以拥有数据
      torch::Tensor t =
          torch::from_blob(rgb.data, {rgb.rows, rgb.cols, 3}, torch::kUInt8)
              .permute({2, 0, 1})  // [3, H, W]
              .clone();

      frames.emplace_back(std::move(t));
    }

    if (frames.empty()) {
      LOG(ERROR) << "OpenCVVideoDecoder got 0 frame: " << path;
      return false;
    }
    return true;
  }

 private:
  bool write_to_temp_file(const std::string& data, std::string& out_path) {
    // 用 mkstemp 生成安全的临时文件
    char tmpl[] =
        "/export/home/wangziyue.28/test/temp/xllm_video_XXXXXX.mp4";  // 后缀仅用于提示；容器里一般也能解
    int fd = mkstemps(tmpl, 4 /*“.mp4”长度*/);
    if (fd == -1) return false;

    // 写入字节
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
