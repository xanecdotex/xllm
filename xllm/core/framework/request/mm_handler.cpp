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

#include "mm_handler.h"

#include <butil/base64.h>
#include <butil/strings/string_number_conversions.h>
#include <glog/logging.h>

#include "common/global_flags.h"
#include "core/util/http_downloader.h"
#include "mm_codec.h"
#include "mm_embedding_handler.h"
#include "mm_input.h"

namespace xllm {

bool MMHandlerBase::process(const MMContent& content,
                            MMInputItem& input,
                            MMPayload& payload,
                            std::string& err_msg) {
  if (!this->load(content, input, payload, err_msg)) {
    LOG(ERROR) << err_msg;
    return false;
  }

  if (!this->decode(input, err_msg)) {
    LOG(ERROR) << err_msg;
    return false;
  }

  return true;
}

bool MMHandlerBase::load_from_dataurl(const std::string& url,
                                      std::string& raw_data,
                                      MMPayload& payload,
                                      const std::string& media_type,
                                      std::string& err_msg) {
  size_t pos = url.find_first_of(';');
  if (pos == std::string::npos) return false;

  butil::StringPiece sub(url, pos + 1);
  pos = sub.find_first_of(',');
  if (pos == std::string::npos) return false;

  butil::StringPiece type(sub, 0, pos);
  butil::StringPiece data(sub, pos + 1);

  if (type == "base64") {
    if (!butil::Base64Decode(data, &raw_data)) {
      err_msg = "invalid " + media_type + " base64 data url: " + url;
      LOG(ERROR) << err_msg;
      return false;
    }
    return true;
  } else if (type == "binary") {
    size_t len = 0;
    bool res = butil::StringToSizeT(data, &len);
    if (res) {
      return payload.get(raw_data, len);
    } else {
      err_msg = "invalid " + media_type + " binary data url: " + url;
      LOG(ERROR) << err_msg;
      return false;
    }
  } else {
    err_msg = "invalid " + media_type + " data url:  " + url;
    LOG(ERROR) << err_msg;
    return false;
  }
}

bool MMHandlerBase::load_from_local(const std::string& url, std::string& data) {
  return false;
}

bool MMHandlerBase::load_from_http(const std::string& url,
                                   std::string& data,
                                   const std::string& media_type,
                                   std::string& err_msg) {
  BRpcDownloader helper_;
  if (!helper_.fetch_data(url, data)) {
    err_msg = "failed to download " + media_type + " from http url: " + url;
    LOG(ERROR) << err_msg;
    return false;
  }
  return true;
}

bool ImageHandler::load(const MMContent& content,
                        MMInputItem& input,
                        MMPayload& payload,
                        std::string& err_msg) {
  input.clear();

  const auto& image_url = content.image_url;
  const auto& url = image_url.url;

  if (url.compare(0, dataurl_prefix_.size(), dataurl_prefix_) ==
      0) {  // data url

    input.type = MMType::IMAGE;
    return this->load_from_dataurl(
        url, input.raw_data, payload, "image", err_msg);
  } else if (url.compare(0, httpurl_prefix_.size(), httpurl_prefix_) ==
             0) {  // http url

    input.type = MMType::IMAGE;
    return this->load_from_http(url, input.raw_data, "image", err_msg);
  } else {
    err_msg = "invalid image url: " + url;
    LOG(ERROR) << err_msg;
    return false;
  }
}

bool ImageHandler::decode(MMInputItem& input, std::string& err_msg) {
  OpenCVImageDecoder decoder;
  if (!decoder.decode(input.raw_data, input.decode_image)) {
    err_msg = "decode image failed";
    return false;
  }
  return true;
}

bool VideoHandler::load(const MMContent& content,
                        MMInputItem& input,
                        MMPayload& payload,
                        std::string& err_msg) {
  input.clear();

  const auto& video_url = content.video_url;
  const auto& url = video_url.url;

  if (url.compare(0, dataurl_prefix_.size(), dataurl_prefix_) ==
      0) {  // data url

    input.type = MMType::VIDEO;
    return this->load_from_dataurl(
        url, input.raw_data, payload, "video", err_msg);
  } else if (url.compare(0, httpurl_prefix_.size(), httpurl_prefix_) ==
             0) {  // http url

    input.type = MMType::VIDEO;
    return this->load_from_http(url, input.raw_data, "video", err_msg);
  } else {
    err_msg = "invalid video url: " + url;
    LOG(ERROR) << err_msg;
    return false;
  }
}

bool VideoHandler::decode(MMInputItem& input, std::string& err_msg) {
  if (FLAGS_use_audio_in_video) {
    FFmpegAudioDecoder audio_decoder;
    if (audio_decoder.decode(
            input.raw_data, input.decode_audio, input.audio_meta)) {
      input.type |= MMType::AUDIO;
    } else {
      err_msg = "decode audio in video failed";
      LOG(ERROR) << err_msg;
      return false;
    }
  }

  FFmpegVideoDecoder decoder;
  if (!decoder.decode(input.raw_data, input.decode_video, input.video_meta)) {
    err_msg = "decode video failed";
    return false;
  }
  return true;
}

bool AudioHandler::load(const MMContent& content,
                        MMInputItem& input,
                        MMPayload& payload,
                        std::string& err_msg) {
  input.clear();

  const auto& audio_url = content.audio_url;
  const auto& url = audio_url.url;

  if (url.compare(0, dataurl_prefix_.size(), dataurl_prefix_) ==
      0) {  // data url

    input.type = MMType::AUDIO;
    return this->load_from_dataurl(
        url, input.raw_data, payload, "audio", err_msg);
  } else if (url.compare(0, httpurl_prefix_.size(), httpurl_prefix_) ==
             0) {  // http url

    input.type = MMType::AUDIO;
    return this->load_from_http(url, input.raw_data, "audio", err_msg);
  } else {
    err_msg = "invalid audio url: " + url;
    LOG(ERROR) << err_msg;
    return false;
  }
}

bool AudioHandler::decode(MMInputItem& input, std::string& err_msg) {
  FFmpegAudioDecoder decoder;
  if (!decoder.decode(input.raw_data, input.decode_audio, input.audio_meta)) {
    err_msg = "decode audio failed";
    return false;
  }
  return true;
}

MMHandlerSet::MMHandlerSet() {
  handlers_["image_url"] = std::make_unique<ImageHandler>();
  handlers_["video_url"] = std::make_unique<VideoHandler>();
  handlers_["audio_url"] = std::make_unique<AudioHandler>();
  handlers_["image_embedding"] =
      std::make_unique<MMEmbeddingHandler>(MMType::IMAGE);
  handlers_["video_embedding"] =
      std::make_unique<MMEmbeddingHandler>(MMType::VIDEO);
  handlers_["audio_embedding"] =
      std::make_unique<MMEmbeddingHandler>(MMType::AUDIO);
}

MMHandlerSet::~MMHandlerSet() {}

bool MMHandlerSet::process(const std::string& type,
                           const MMContent& content,
                           MMInputItem& input,
                           MMPayload& payload,
                           std::string& err_msg) {
  auto itor = handlers_.find(type);
  if (itor == handlers_.end()) {
    err_msg = "unsupported mm type: " + type;
    return false;
  }

  auto& handler = itor->second;
  return handler->process(content, input, payload, err_msg);
}

}  // namespace xllm
