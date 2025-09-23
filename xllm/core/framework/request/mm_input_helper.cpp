/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mm_input_helper.h"

#include <brpc/channel.h>
#include <brpc/controller.h>
#include <bthread/rwlock.h>
#include <fcntl.h>
#include <glog/logging.h>
#include <linux/memfd.h>
#include <sys/mman.h>
#include <unistd.h>

#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>

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

class FileDownloadHelper {
 public:
  FileDownloadHelper() {}
  ~FileDownloadHelper() {}
  std::string parse_url(const std::string& url) {
    size_t scheme_end = url.find("://");
    if (scheme_end == std::string::npos) {
      LOG(ERROR)
          << "Error: Invalid URL, missing protocol (http:// or https://)";
    }
    size_t host_start = scheme_end + 3;
    size_t path_pos = url.find('/', host_start);
    if (path_pos == std::string::npos) {
      LOG(ERROR) << "Error: No path in URL";
    }
    return url.substr(host_start, path_pos - host_start);
  }

  std::shared_ptr<brpc::Channel> get_channel(const std::string& host) {
    {
      bthread::RWLockRdGuard rd_guard(instance_channel_map_mutex_);
      auto it = channels_.find(host);
      if (it != channels_.end()) {
        return it->second;
      }
    }
    bthread::RWLockWrGuard wr_guard(instance_channel_map_mutex_);
    auto it = channels_.find(host);
    if (it != channels_.end()) {
      return it->second;
    }

    brpc::ChannelOptions option;
    option.protocol = brpc::PROTOCOL_HTTP;
    option.connection_type = brpc::CONNECTION_TYPE_POOLED;
    option.max_retry = 3;
    auto channel = std::make_shared<brpc::Channel>();
    if (channel->Init(host.c_str(), &option) != 0) {
      LOG(ERROR) << "fail to init channel for " << host;
      return nullptr;
    }
    channels_[host] = channel;
    return channel;
  }

  bool download_data(const std::string& host,
                     const std::string& url,
                     std::string& data) {
    brpc::Controller cntl;
    cntl.http_request().uri() = url;
    cntl.set_timeout_ms(2000);
    auto channel = get_channel(host);
    if (!channel) {
      LOG(ERROR) << "Channel is null";
      return false;
    }
    channel->CallMethod(nullptr, &cntl, nullptr, nullptr, nullptr);
    if (cntl.Failed()) {
      LOG(ERROR) << "Request failed: " << cntl.ErrorText();
      return false;
    }

    if (cntl.http_response().status_code() != 200) {
      LOG(ERROR) << "HTTP error: " << cntl.http_response().status_code();
      return false;
    }

    const butil::IOBuf& io = cntl.response_attachment();
    data = io.to_string();
    return true;
  }

  bool fetch_data(const std::string& url, std::string& data) {
    // parse url
    std::string host = parse_url(url);
    // fetch data
    return download_data(host, url, data);
  }

 private:
  bthread::RWLock instance_channel_map_mutex_;
  inline static std::unordered_map<std::string, std::shared_ptr<brpc::Channel>>
      channels_;
};

class OpenCVVideoDecoder {
 public:
  struct LocalReader : cv::IStreamReader {
    const unsigned char* p;
    long long n;
    long long pos = 0;
    LocalReader(const unsigned char* d, size_t s) : p(d), n((long long)s) {}
    long long read(char* buf, long long sz) override {
      if (!buf || sz <= 0) return 0;
      long long rem = n - pos;
      if (rem <= 0) return 0;
      long long k = std::min(rem, sz);
      std::memcpy(buf, p + pos, (size_t)k);
      pos += k;
      return k;
    }
    long long seek(long long off, int orig) override {
      long long np = pos;
      if (orig == SEEK_SET)
        np = off;
      else if (orig == SEEK_CUR)
        np = pos + off;
      else if (orig == SEEK_END)
        np = n + off;
      else
        return -1;
      if (np < 0) np = 0;
      if (np > n) np = n;
      pos = np;
      return pos;
    }
  };

  bool decode(const std::string& raw_data, torch::Tensor& t, double& fps) {
    cv::Ptr<cv::IStreamReader> reader = cv::makePtr<LocalReader>(
        reinterpret_cast<const unsigned char*>(raw_data.data()),
        raw_data.size());

    cv::VideoCapture cap(reader, cv::CAP_FFMPEG, std::vector<int>{});
    if (!cap.isOpened()) {
      LOG(INFO) << " opencv video decode failed";
      return false;
    } else {
      double video_fps = cap.get(cv::CAP_PROP_FPS);
      if (video_fps <= 0 || std::isnan(video_fps)) video_fps = 30.0;

      std::vector<torch::Tensor> frames;
      int est = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
      if (est < 0) est = 0;
      frames.reserve(est + 1);

      cv::Mat image;
      while (cap.read(image)) {
        if (image.empty()) break;
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        torch::Tensor tensor = torch::from_blob(
            image.data, {image.rows, image.cols, 3}, torch::kUInt8);
        frames.emplace_back(tensor.permute({2, 0, 1}).clone());
      }

      if (!frames.empty()) {
        t = torch::stack(frames, 0);  // [T,C,H,W]
        fps = video_fps;
      } else {
        LOG(INFO) << "opencv video decode got 0 frame ";
        return false;
      }
    }
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
    data.clear();
    std::string path = url;
    if (path.rfind("file://", 0) == 0) {
      path.erase(0, 7);
      if (path.rfind("localhost/", 0) == 0) path.erase(0, 10);
    } else if (path.rfind("file:", 0) == 0) {
      path.erase(0, 5);
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    data.assign(std::istreambuf_iterator<char>(in),
                std::istreambuf_iterator<char>());
    return static_cast<bool>(in) || in.eof();
  }

  bool load_from_http(const std::string& url, std::string& data) {
    return helper_.fetch_data(url, data);
  }

  std::string dataurl_prefix_{"data:image"};
  std::string httpurl_prefix_{"http"};

 private:
  FileDownloadHelper helper_;
};

class ImageHandler : public Handler {
 public:
  ImageHandler() {}

  virtual bool load(const proto::MMInputData& msg, MMInputItem& input) {
    input.clear();

    const auto& image_url = msg.image_url();
    const auto& url = image_url.url();

    if (url.compare(0, dataurl_prefix_.size(), dataurl_prefix_) ==
        0) {  // data url

      input.type_ = MMType::IMAGE;
      return this->load_from_dataurl(url, input.raw_data_);
    } else if (url.compare(0, httpurl_prefix_.size(), httpurl_prefix_) ==
               0) {  // http url

      input.type_ = MMType::IMAGE;
      return this->load_from_http(url, input.raw_data_);
    }
  }

  virtual bool decode(MMInputItem& input) {
    OpenCVImageDecoder decoder;
    return decoder.decode(input.raw_data_, input.decode_data_);
  }
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

    if (url.compare(0, dataurl_prefix_.size(), dataurl_prefix_) == 0) {
      return this->load_from_dataurl(url, input.raw_data_);
    } else if (url.compare(0, http_prefix1_.size(), http_prefix1_) == 0 ||
               url.compare(0, http_prefix2_.size(), http_prefix2_) == 0) {
      return this->load_from_http(url, input.raw_data_);
    } else if (!url.empty()) {
      return this->load_from_local(url, input.raw_data_);
    } else {
      return false;
    }
  }

  bool decode(MMInputItem& input) override {
    OpenCVVideoDecoder decoder;
    return decoder.decode(input.raw_data_, input.decode_data_, input.fps_);
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
  std::unordered_map<std::string, std::unique_ptr<Handler>> handlers_;
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
