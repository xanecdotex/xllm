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

#include "clip_image_processor.h"

namespace xllm {

torch::Tensor ImageProcessor::resize(const torch::Tensor& image,
                                     const std::vector<int64_t>& size,
                                     int resample,
                                     bool antialias) {
  if (image.dim() != 3) {
    throw std::invalid_argument("Input image must be a 3D tensor (C x H x W).");
  }
  auto options = torch::nn::functional::InterpolateFuncOptions()
                     .size(size)
                     .align_corners(false)
                     .antialias(antialias);
  switch (resample) {
    case 1:
      options.mode(torch::kNearest);
      break;
    case 2:
      options.mode(torch::kBilinear);
      break;
    case 3:
      options.mode(torch::kBicubic);
      break;
    default:
      throw std::invalid_argument(
          "Invalid resample value. Must be one of 1, 2, or 3.");
  }
  return torch::nn::functional::interpolate(image.unsqueeze(0), options)
      .squeeze(0)
      .clamp(0, 255)
      .to(torch::kUInt8);
}

torch::Tensor ImageProcessor::centerCrop(const torch::Tensor& image,
                                         const std::pair<int, int>& cropSize) {
  if (image.dim() != 3) {
    throw std::runtime_error(
        "Input image must be a 3-dimensional tensor in (C, H, W) format.");
  }

  int cropHeight = cropSize.first;
  int cropWidth = cropSize.second;
  int origHeight = image.size(1);
  int origWidth = image.size(2);

  int top = (origHeight - cropHeight) / 2;
  int bottom = top + cropHeight;
  int left = (origWidth - cropWidth) / 2;
  int right = left + cropWidth;

  if (top >= 0 && bottom <= origHeight && left >= 0 && right <= origWidth) {
    return image.index({torch::indexing::Slice(),
                        torch::indexing::Slice(top, bottom),
                        torch::indexing::Slice(left, right)});
  }

  int newHeight = std::max(cropHeight, origHeight);
  int newWidth = std::max(cropWidth, origWidth);
  auto paddedImage =
      torch::zeros({image.size(0), newHeight, newWidth}, image.options());

  int topPad = (newHeight - origHeight + 1) / 2;
  int leftPad = (newWidth - origWidth + 1) / 2;

  paddedImage.index_put_({torch::indexing::Slice(),
                          torch::indexing::Slice(topPad, topPad + origHeight),
                          torch::indexing::Slice(leftPad, leftPad + origWidth)},
                         image);

  top = (newHeight - cropHeight) / 2;
  bottom = top + cropHeight;
  left = (newWidth - cropWidth) / 2;
  right = left + cropWidth;

  return paddedImage.index({torch::indexing::Slice(),
                            torch::indexing::Slice(top, bottom),
                            torch::indexing::Slice(left, right)});
}

torch::Tensor ImageProcessor::rescale(const torch::Tensor& image,
                                      double scale) {
  return image * scale;
}

torch::Tensor ImageProcessor::normalize(const torch::Tensor& image,
                                        const std::vector<double>& mean,
                                        const std::vector<double>& std) {
  if (image.dim() != 3) {
    throw std::runtime_error(
        "Input image must be a 3-dimensional tensor in (C, H, W) format.");
  }

  int numChannels = image.size(0);
  if (mean.size() != numChannels || std.size() != numChannels) {
    throw std::runtime_error(
        "Mean and std vectors must have the same number "
        "of elements as the number of channels in the "
        "image.");
  }

  auto result = image;
  if (!image.is_floating_point()) {
    result = image.to(torch::kFloat32);
  }

  auto dtype = image.dtype();
  auto device = image.device();
  auto options = torch::dtype(dtype).device(device);

  auto m_tensor = torch::tensor(mean, options).reshape({-1, 1, 1});
  auto s_tensor = torch::tensor(std, options).reshape({-1, 1, 1});

  result = result.sub(m_tensor);
  return result.div_(s_tensor);
}

torch::Tensor ImageProcessor::init_frames(const torch::Tensor& video) {
  // linspace
  auto video_frames = video.unbind(0);
  int total_num_frames = static_cast<int>(video_frames.size());
  int nframes_len = 32;
  auto idx = torch::linspace(0,
                             static_cast<double>(total_num_frames - 1),
                             nframes_len,
                             torch::dtype(torch::kFloat32))
                 .round()
                 .to(torch::kLong)
                 .contiguous();
  const int64_t* idx_ptr = idx.data_ptr<int64_t>();
  std::vector<torch::Tensor> picked_frames;
  picked_frames.reserve(nframes_len);
  for (int i = 0; i < nframes_len; ++i) {
    int64_t k = static_cast<int64_t>(idx_ptr[i]);
    picked_frames.emplace_back(video_frames[k]);
  }
  video_frames = std::move(picked_frames);
  return torch::stack(video_frames, 0);
}

torch::Tensor ImageProcessor::sample_frames(const torch::Tensor& video,
                                            double video_fps,
                                            int temporal_patch_size,
                                            int min_frames,
                                            int max_frames,
                                            int num_frames,
                                            double set_fps) {
  auto video_frames = video.unbind(0);
  if (set_fps > 0.0 && num_frames > 0) {
    LOG(FATAL) << "num_frames and fps are mutually exclusive arguments, please "
                  "use only one!";
  }

  double fps = set_fps;

  int total_num_frames = static_cast<int>(video_frames.size());

  if (num_frames > 0) {
    double double_num_frames =
        std::round(static_cast<double>(num_frames) / temporal_patch_size) *
        temporal_patch_size;
    num_frames = static_cast<int>(double_num_frames);
  } else if (fps > 0.0) {
    if (video_fps <= 0.0) {
      LOG(FATAL)
          << "Asked to sample `fps` frames per second but no video metadata "
             "was provided which is required when sampling with `fps`. ";
    }

    max_frames =
        (std::min(max_frames, total_num_frames) / temporal_patch_size) *
        temporal_patch_size;
    double double_num_frames =
        static_cast<double>(total_num_frames) / video_fps * fps;
    double_num_frames = std::min(
        std::min(std::max(double_num_frames, static_cast<double>(min_frames)),
                 static_cast<double>(max_frames)),
        static_cast<double>(total_num_frames));
    double_num_frames = std::floor(double_num_frames / temporal_patch_size) *
                        temporal_patch_size;

    num_frames = static_cast<int>(double_num_frames);
  }

  if (num_frames > total_num_frames) {
    LOG(FATAL) << "Video can't be sampled. The inferred num_frames="
               << num_frames << " exceeds total_num_frames=" << total_num_frames
               << ".";
  }

  if (num_frames > 0) {
    LOG(INFO) << num_frames;
    std::vector<torch::Tensor> picked_frames;
    picked_frames.reserve(num_frames);
    for (int i = 0; i < num_frames; ++i) {
      int64_t k = static_cast<int64_t>(
          (static_cast<int64_t>(i) * total_num_frames) / num_frames);
      if (k >= total_num_frames) k = total_num_frames - 1;
      picked_frames.push_back(video_frames[k]);
    }
    return torch::stack(picked_frames, 0);
  } else {
    return video;
  }
}

}  // namespace xllm
