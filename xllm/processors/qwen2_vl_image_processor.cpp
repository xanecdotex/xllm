#include "qwen2_vl_image_processor.h"

namespace xllm {

namespace {

using Size = std::pair<int, int>;

static inline int64_t ceil_to_multiple(int64_t x, int64_t f) {
  return (x + f - 1) / f * f;
}
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
  LOG(INFO) << "video size:" << videos.size();
  if (images.empty() && videos.empty()) {
    LOG(ERROR) << "no image/video tensor found.";
    return false;
  }

  // 仅图片
  if (!images.empty() && videos.empty()) {
    return this->process_images(images, datas);
  }

  // 仅视频
  if (images.empty() && !videos.empty()) {
    return this->process_videos(videos, datas);
  }
  // MMData img_data, vid_data;
  //   if (!this->process_images(images, img_data)) return false;
  //   if (!this->process_videos(videos, vid_data)) return false;
  // auto img_vals = img_data["pixel_values"].toTensor();
  // auto vid_vals = vid_data["pixel_values"].toTensor();
  // auto all_vals = torch::cat({img_vals, vid_vals}, 0);

  // auto img_thw  = img_data["image_grid_thw"].toTensor();
  // auto vid_thw  = vid_data["image_grid_thw"].toTensor();
  // auto all_thw  = torch::cat({img_thw, vid_thw}, 0);

  // datas = MMData(MMType::MIXED, {
  //     {"pixel_values",   all_vals},
  //     {"image_grid_thw", all_thw}
  // });
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
                                           MMData& mm_datas) {
  std::vector<torch::Tensor> pixel_values;
  std::vector<int64_t> grids;

  for (auto& vid : videos) {
    if (!this->process_video(vid, pixel_values, grids)) {
      return false;
    }
  }

  auto values = torch::cat(pixel_values);  // [sum(Nv), C*Tps*P*P]
  auto thw = torch::tensor(grids).clone().reshape({-1, 3});  // [Nv, 3]

  mm_datas = MMData(MMType::VIDEO,
                    {{"image_grid_thw", thw}, {"pixel_values", values}});
  return true;
}

bool Qwen2VLImageProcessor::process_video(
    torch::Tensor video,  // [T, C, H, W], dtype = uint8 or float
    std::vector<torch::Tensor>& pixel_values,  // push 一个视频的所有 patch
    std::vector<int64_t>& grids) {             // push [grid_t, grid_h, grid_w]
  // 校验形状
  TORCH_CHECK(video.dim() == 4, "video must be TCHW");
  auto sizes = video.sizes();
  int64_t T = sizes[0];
  int64_t C = sizes[1];
  int64_t H = sizes[2];
  int64_t W = sizes[3];

  // 1) smart_resize：得到目标尺寸
  int64_t resized_h = H, resized_w = W;
  if (do_resize_) {
    auto size = smart_resize(H,
                             W,
                             patch_size_ * merge_size_,  // factor
                             size_["shortest_edge"],     // min edge
                             size_["longest_edge"]);     // max edge
    if (!size) return false;
    std::tie(resized_h, resized_w) = *size;
  }

  // 2) 统一 resize 所有帧（BICUBIC + antialias）
  //    将 [T, C, H, W] 当成 batch 维度 T 的 NCHW 做插值
  if (do_resize_ && (resized_h != H || resized_w != W)) {
    using F = torch::nn::functional::InterpolateFuncOptions;
    video = torch::nn::functional::interpolate(
        video,
        F().size(std::vector<int64_t>{resized_h, resized_w})
            .mode(torch::kBicubic)
            .align_corners(false)
            .antialias(true));
  }

  // 3) 归一化 / 缩放（与图片保持一致）
  //    若输入是 uint8，先转 float
  if (!video.is_floating_point()) {
    video = video.to(torch::kFloat);
  }
  if (do_normalize_) {
    video = this->normalize(video, image_mean_, image_std_);  // 逐通道
  }
  if (do_rescale_) {
    video = this->rescale(video, rescale_factor_);  // e.g., 1/255
  }

  // 4) 时间维补齐到 temporal_patch_size_ 的倍数（重复最后一帧）
  const int64_t tps = temporal_patch_size_;
  int64_t T_pad = ceil_to_multiple(T, tps);
  if (T_pad > T) {
    auto last = video.index({T - 1}).unsqueeze(0);    // [1, C, H, W]
    auto repeat = last.repeat({T_pad - T, 1, 1, 1});  // [T_pad-T, C, H, W]
    video = torch::cat({video, repeat}, /*dim=*/0);   // [T_pad, C, H, W]
    T = T_pad;
  }

  // 5) 切块（严格对应你 process_image 的张量变换）
  // grid 尺寸（空间）
  TORCH_CHECK(resized_h % patch_size_ == 0 && resized_w % patch_size_ == 0,
              "H/W must be divisible by patch_size");
  int64_t grid_h = resized_h / patch_size_;
  int64_t grid_w = resized_w / patch_size_;

  // grid 尺寸（时间）
  TORCH_CHECK(T % tps == 0, "T must be divisible by temporal_patch_size_");
  int64_t grid_t = T / tps;

  // reshape / permute / flatten
  // 目标： [grid_t * grid_h * grid_w, C * tps * P * P]
  // 步骤与 process_image 完全一致，只是把原先的 repeats 换成真实帧
  video = video.contiguous();  // 之后的 view/permute 更安全
  auto P = patch_size_;
  auto M = merge_size_;

  // [T, C, H, W] → [grid_t, tps, C, (grid_h/M), M, P, (grid_w/M), M, P]
  auto reshaped =
      video.view({grid_t, tps, C, grid_h / M, M, P, grid_w / M, M, P});

  // permute 次序与图片版保持一致：
  // {0, 3, 6, 4, 7, 2, 1, 5, 8}
  // → t, (gh/M), (gw/M), M, M, C, tps, P, P
  auto perm = reshaped.permute({0, 3, 6, 4, 7, 2, 1, 5, 8});

  // 展平： [grid_t * grid_h * grid_w, C * tps * P * P]
  auto patches = perm.reshape({grid_t * grid_h * grid_w, C * tps * P * P});

  // 6) 记录输出
  pixel_values.emplace_back(patches);  // 一个视频的所有 patch
  grids.insert(grids.end(), {grid_t, grid_h, grid_w});
  return true;
}

}  // namespace xllm
