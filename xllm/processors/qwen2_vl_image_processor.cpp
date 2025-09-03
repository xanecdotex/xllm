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

#include "qwen2_vl_image_processor.h"

namespace xllm {

namespace {

using Size = std::pair<int, int>;

std::optional<Size> smart_resize(int height,
                                 int width,
                                 int factor = 28,
                                 int min_pixels = 56 * 56,
                                 int max_pixels = 14 * 14 * 4 * 1280) {
  if (height < factor || width < factor) {
    LOG(ERROR) << "Height or width must be larger than factor";
    return std::nullopt;
  }

  if (static_cast<double>(std::max(height, width)) / std::min(height, width) >
      200) {
    LOG(ERROR) << "Absolute aspect ratio must be smaller than 200";
    return std::nullopt;
  }

  int h_bar =
      static_cast<int>(std::round(height / static_cast<double>(factor))) *
      factor;
  int w_bar =
      static_cast<int>(std::round(width / static_cast<double>(factor))) *
      factor;

  if (h_bar * w_bar > max_pixels) {
    double beta = std::sqrt((height * width) / static_cast<double>(max_pixels));
    h_bar = static_cast<int>(
                std::floor(height / beta / static_cast<double>(factor))) *
            factor;
    w_bar = static_cast<int>(
                std::floor(width / beta / static_cast<double>(factor))) *
            factor;
  } else if (h_bar * w_bar < min_pixels) {
    double beta = std::sqrt(min_pixels / static_cast<double>(height * width));
    h_bar = static_cast<int>(
                std::ceil(height * beta / static_cast<double>(factor))) *
            factor;
    w_bar = static_cast<int>(
                std::ceil(width * beta / static_cast<double>(factor))) *
            factor;
  }

  return std::make_pair(h_bar, w_bar);
}
}  // namespace

Qwen2VLImageProcessor::Qwen2VLImageProcessor(const ModelArgs& args) {
  image_mean_ = args.mm_image_normalize_mean();
  image_std_ = args.mm_image_normalize_std();

  min_pixels_ = args.mm_image_min_pixels();
  max_pixels_ = args.mm_image_max_pixels();

  patch_size_ = args.mm_image_patch_size();
  temporal_patch_size_ = args.mm_image_temporal_patch_size();

  merge_size_ = args.mm_image_merge_size();
  size_ = {{"longest_edge", 12845056}, {"shortest_edge", 3136}};

  min_frames_ = args.mm_video_min_frames();
  max_frames_ = args.mm_video_max_frames();

  // fuse image mean/std and rescale_factor
  if (do_rescale_ && do_normalize_) {
    for (auto& item : image_mean_) {
      item = item * (1.0 / rescale_factor_);
    }

    for (auto& item : image_std_) {
      item = item * (1.0 / rescale_factor_);
    }

    do_rescale_ = false;
  }
}

bool Qwen2VLImageProcessor::process(const MMInput& inputs, MMData& datas) {
  std::vector<torch::Tensor> images = inputs.get_decode_data(MMType::IMAGE);
  std::vector<torch::Tensor> videos = inputs.get_decode_data(MMType::VIDEO);
  std::vector<double> video_fps_list = inputs.get_video_fps(MMType::VIDEO);

  if (images.empty() && (videos.empty() || video_fps_list.empty())) {
    LOG(ERROR) << "no image/video tensor found.";
    return false;
  }

  if (!images.empty() && videos.empty()) {
    if (!this->process_images(images, datas)) {
      LOG(ERROR) << " process image failed.";
      return false;
    }
  }

  if (images.empty() && !videos.empty()) {
    if (!this->process_videos(videos, video_fps_list, datas)) {
      LOG(ERROR) << " process video failed.";
      return false;
    }
  }

  return true;
}

bool Qwen2VLImageProcessor::process_images(std::vector<torch::Tensor> images,
                                           MMData& mm_datas) {
  std::vector<torch::Tensor> pixel_values;
  std::vector<int64_t> grids;

  for (const auto& img : images) {
    if (!this->process_image(img, pixel_values, grids)) {
      return false;
    }
  }

  auto values = torch::cat(pixel_values);
  auto thw = torch::tensor(grids);

  thw = thw.clone().reshape({-1, 3});
  mm_datas = std::move(MMData(
      MMType::IMAGE, {{"image_grid_thw", thw}, {"pixel_values", values}}));

  return true;
}

bool Qwen2VLImageProcessor::process_image(
    torch::Tensor image,
    std::vector<torch::Tensor>& pixel_values,
    std::vector<int64_t>& grids) {
  auto shape = image.sizes();

  auto resized_height = shape[1];
  auto resized_width = shape[2];

  // do_convert_rgb

  // resize
  if (do_resize_) {
    auto size = smart_resize(resized_height,
                             resized_width,
                             patch_size_ * merge_size_,
                             size_["shortest_edge"],
                             size_["longest_edge"]);
    if (!size) {
      return false;
    }

    std::tie(resized_height, resized_width) = *size;
    image =
        this->resize(image, {resized_height, resized_width}, resample_, false);
  }

  // normalize
  if (do_normalize_) {
    image = this->normalize(image, image_mean_, image_std_);
  }

  // rescale
  if (do_rescale_) {
    image = this->rescale(image, rescale_factor_);
  }

  auto patches = torch::stack({image}, 0);
  auto repeats =
      patches[-1].unsqueeze(0).repeat({temporal_patch_size_ - 1, 1, 1, 1});
  patches = torch::cat({patches, repeats}, 0);
  shape = patches.sizes();

  auto channel = shape[1];
  auto grid_t = shape[0] / temporal_patch_size_;

  auto grid_h = resized_height / patch_size_;
  auto grid_w = resized_width / patch_size_;

  patches = patches.view({grid_t,
                          temporal_patch_size_,
                          channel,
                          grid_h / merge_size_,
                          merge_size_,
                          patch_size_,
                          grid_w / merge_size_,
                          merge_size_,
                          patch_size_});

  patches = patches.permute({0, 3, 6, 4, 7, 2, 1, 5, 8});
  patches = patches.reshape(
      {grid_t * grid_h * grid_w,
       channel * temporal_patch_size_ * patch_size_ * patch_size_});

  pixel_values.emplace_back(patches);
  grids.insert(grids.end(), {grid_t, grid_h, grid_w});

  return true;
}

bool Qwen2VLImageProcessor::process_videos(std::vector<torch::Tensor> videos,
                                           std::vector<double> video_fps_list,
                                           MMData& mm_datas) {
  std::vector<torch::Tensor> pixel_values;
  std::vector<int64_t> grids;

  const size_t video_size = videos.size();
  for (size_t i = 0; i < video_size; ++i) {
    auto& vid = videos[i];
    double fps = video_fps_list[i];
    if (!this->process_video(vid, fps, pixel_values, grids)) {
      return false;
    }
  }

  auto values = torch::cat(pixel_values);
  auto thw = torch::tensor(grids).clone().reshape({-1, 3});
  const size_t num_videos = videos.size();

  std::vector<double> second_per_grid;
  second_per_grid.reserve(num_videos);
  for (size_t i = 0; i < num_videos; ++i) {
    double fps = 2.0;
    double seconds_per_grid = static_cast<double>(temporal_patch_size_ / fps);
    second_per_grid.push_back(seconds_per_grid);
  }
  auto opts = torch::TensorOptions().dtype(torch::kFloat32);
  auto second_per_grid_ts =
      torch::tensor(second_per_grid, opts);  // [num_videos]

  mm_datas = MMData(MMType::VIDEO,
                    {{"video_grid_thw", thw},
                     {"pixel_values_videos", values},
                     {"second_per_grid_ts", second_per_grid_ts}});
  return true;
}

bool Qwen2VLImageProcessor::process_video(
    torch::Tensor origin_video,
    double fps,
    std::vector<torch::Tensor>& pixel_values,
    std::vector<int64_t>& grids) {
  if (origin_video.dim() != 4) {
    LOG(FATAL) << "video must be TCHW";
  }
  torch::Tensor video =
      this->init_frames(origin_video);  // default sample to 32 frames

  if (do_sample_frame_) {
    torch::Tensor video = this->sample_frames(
        origin_video, fps, temporal_patch_size_, min_frames_, max_frames_);
  }

  auto shape = video.sizes();
  auto time_len = shape[0];
  auto channel = shape[1];
  auto resized_height = shape[2];
  auto resized_width = shape[3];

  if (do_resize_) {
    auto size = smart_resize(resized_height,
                             resized_width,
                             patch_size_ * merge_size_,
                             size_["shortest_edge"],
                             size_["longest_edge"]);
    if (!size) {
      return false;
    }
    std::tie(resized_height, resized_width) = *size;
  }

  std::vector<torch::Tensor> out_frames;
  out_frames.reserve(time_len);
  // for each frame
  auto frames = video.unbind(0);
  for (auto& frame : frames) {
    // resize
    if (do_resize_)
      frame =
          this->resize(frame, {resized_height, resized_width}, resample_, true);
    // normalize
    if (do_normalize_) frame = this->normalize(frame, image_mean_, image_std_);
    // rescale
    if (do_rescale_) frame = this->rescale(frame, rescale_factor_);
    out_frames.push_back(frame);
  }

  auto out_video = torch::stack(out_frames);  // [T,C,H,W]

  auto pad_t = (temporal_patch_size_ - (time_len % temporal_patch_size_)) %
               temporal_patch_size_;
  if (pad_t > 0) {
    auto last =
        out_video.index({time_len - 1}).unsqueeze(0).repeat({pad_t, 1, 1, 1});
    out_video = torch::cat({out_video, last}, 0);
  }

  shape = out_video.sizes();
  auto grid_h = resized_height / patch_size_;
  auto grid_w = resized_width / patch_size_;
  auto grid_t = shape[0] / temporal_patch_size_;

  out_video = out_video.contiguous();

  auto patches = out_video.view({grid_t,
                                 temporal_patch_size_,
                                 channel,
                                 grid_h / merge_size_,
                                 merge_size_,
                                 patch_size_,
                                 grid_w / merge_size_,
                                 merge_size_,
                                 patch_size_});

  patches = patches.permute({0, 3, 6, 4, 7, 2, 1, 5, 8});
  patches = patches.reshape(
      {grid_t * grid_h * grid_w,
       channel * temporal_patch_size_ * patch_size_ * patch_size_});

  pixel_values.emplace_back(patches);
  grids.insert(grids.end(), {grid_t, grid_h, grid_w});
  return true;
}

}  // namespace xllm
