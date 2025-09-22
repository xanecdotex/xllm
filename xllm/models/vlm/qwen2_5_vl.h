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

#pragma once

#include <atb/atb_infer.h>
#include <c10/core/ScalarType.h>
#include <torch/torch.h>

#include <unordered_map>

#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/layers/npu/llm_head.h"
#include "core/layers/npu/qwen2_5_vision_encoder_layer.h"
#include "core/layers/npu/qwen2_decoder_layer.h"
#include "core/layers/npu/rms_norm.h"
#include "models/llm/qwen2.h"
#include "models/model_registry.h"
#include "processors/input_processor.h"
#include "processors/qwen2_vl_image_processor.h"
#include "xllm_kernels/core/include/atb_speed/log.h"

namespace xllm::hf {

#define PrintTensor(tensor) print_tensor(tensor, #tensor, 10, true, false);

class Qwen2_5_VLInputProcessor : public InputProcessor {
  enum class TokenType {
    INVALID,
    IMAGE,
    VIDEO,
  };

 public:
  Qwen2_5_VLInputProcessor(const ModelArgs& args) {
    merge_size_ = args.mm_image_merge_size();
  }

  void process(std::string& prompt, const MMData& mm_data) override {
    torch::Tensor image_grid_thw;
    if (auto res = mm_data.get<torch::Tensor>("image_grid_thw"))
      image_grid_thw = res.value();

    torch::Tensor video_grid_thw;
    if (auto res = mm_data.get<torch::Tensor>("video_grid_thw"))
      video_grid_thw = res.value();

    if (!image_grid_thw.defined() && !video_grid_thw.defined()) return;

    auto merge_length = merge_size_ * merge_size_;
    int total_image_token = 0;
    if (image_grid_thw.defined()) {
      auto count = image_grid_thw.sizes()[0];
      for (int idx = 0; idx < count; ++idx)
        total_image_token +=
            image_grid_thw[idx].prod().item<int>() / merge_length;
    }

    int total_video_token = 0;
    if (video_grid_thw.defined()) {
      auto count = video_grid_thw.sizes()[0];
      for (int idx = 0; idx < count; ++idx)
        total_video_token +=
            video_grid_thw[idx].prod().item<int>() / merge_length;
    }

    size_t total_token_len = total_image_token * image_token_.size() +
                             total_video_token * video_token_.size();
    std::string data;
    data.reserve(prompt.size() + total_token_len);

    int image_index = 0;
    int video_index = 0;

    const torch::Tensor* grid_thw = nullptr;
    const std::string* token = nullptr;
    int* index = 0;

    size_t begin = 0;
    auto pair = find_vision_token(prompt, begin);

    while (pair.second != std::string::npos) {
      data.append(prompt, begin, pair.second - begin);

      if (pair.first == TokenType::IMAGE) {
        grid_thw = &image_grid_thw;
        token = &image_token_;
        index = &image_index;
      } else if (pair.first == TokenType::VIDEO) {
        grid_thw = &video_grid_thw;
        token = &video_token_;
        index = &video_index;
      } else {
        assert(false);
      }

      auto token_num = (*grid_thw)[(*index)].prod().item<int>() / merge_length;
      while (token_num--) data.append(*token);

      ++(*index);
      begin = pair.second + token->size();
      pair = find_vision_token(prompt, begin);
    }

    if (begin < prompt.size()) data.append(prompt, begin, std::string::npos);

    prompt = std::move(data);
  }

 private:
  std::pair<TokenType, size_t> find_vision_token(const std::string& prompt,
                                                 size_t begin) {
    auto img_pos = prompt.find(image_token_, begin);
    auto vid_pos = prompt.find(video_token_, begin);

    if (img_pos == std::string::npos && vid_pos == std::string::npos)
      return {TokenType::INVALID, std::string::npos};
    else if (vid_pos == std::string::npos)
      return {TokenType::IMAGE, img_pos};
    else if (img_pos == std::string::npos)
      return {TokenType::VIDEO, vid_pos};
    else
      return img_pos < vid_pos ? std::make_pair(TokenType::IMAGE, img_pos)
                               : std::make_pair(TokenType::VIDEO, vid_pos);
  }

 private:
  const std::string image_token_ = "<|image_pad|>";
  const std::string video_token_ = "<|video_pad|>";

  int merge_size_ = 0;
};

class Qwen2_5_VisionBlockImpl : public torch::nn::Module {
 public:
  Qwen2_5_VisionBlockImpl(const ModelContext& context) {
    // register submodules
    encoder_layer_ =
        register_module("encoder_layer", Qwen2_5VisionEncoder(context));
  }

  torch::Tensor forward(torch::Tensor& x,
                        torch::Tensor& m_cos_pos,
                        torch::Tensor& m_sin_pos,
                        torch::Tensor& cu_seq_len,
                        std::vector<int>& cu_seq_len_vec,
                        ModelInputParams& input_params,
                        int node_id) {
    return encoder_layer_(x,
                          m_cos_pos,
                          m_sin_pos,
                          cu_seq_len,
                          cu_seq_len_vec,
                          input_params,
                          node_id);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    encoder_layer_->load_state_dict(state_dict);
  }

  void verify_loaded_weights(const std::string& prefix) const {
    encoder_layer_->verify_loaded_weights();
  }
  void merge_loaded_weights() { encoder_layer_->merge_loaded_weights(); }

 private:
  Qwen2_5VisionEncoder encoder_layer_{nullptr};
};
TORCH_MODULE(Qwen2_5_VisionBlock);

class Qwen2_5_VisionPatchEmbedImpl : public torch::nn::Module {
 public:
  Qwen2_5_VisionPatchEmbedImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();

    auto in_features = model_args.mm_num_channels() *
                       model_args.mm_temporal_patch_size() *
                       model_args.mm_patch_size() * model_args.mm_patch_size();

    auto out_features = model_args.mm_hidden_size();

    proj_ = register_module(
        "proj",
        torch::nn::Linear(
            torch::nn::LinearOptions(in_features, out_features).bias(false)));

    proj_->weight.set_data(proj_->weight.to(options));
  }

  torch::Tensor forward(torch::Tensor x) { return proj_(x); }

  void load_state_dict(const StateDict& state_dict) {
    auto weight = state_dict.get_tensor("proj.weight");
    if (weight.defined()) {
      weight = weight.reshape({weight.size(0), -1});
      DCHECK_EQ(proj_->weight.sizes(), weight.sizes())
          << "proj weight size mismatch for " << name();
      proj_->weight.data().copy_(weight);
      proj_weight_loaded_ = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(proj_weight_loaded_)
        << "weight is not loaded for " << prefix + "proj.weight";
  }

 private:
  bool proj_weight_loaded_ = false;
  torch::nn::Linear proj_{nullptr};
};
TORCH_MODULE(Qwen2_5_VisionPatchEmbed);

class Qwen2_5_VisionRotaryEmbeddingImpl : public torch::nn::Module {
 public:
  Qwen2_5_VisionRotaryEmbeddingImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();

    dim_ = model_args.mm_head_dim() / 2;
    theta_ = 10000.0;

    auto opts = options.dtype(torch::kFloat32);
    auto inv_freq =
        1.0 / torch::pow(theta_, torch::arange(0, dim_, 2, opts) / dim_);
    inv_freq_ = register_buffer("inv_freq", inv_freq);
  }

  void update_freqs_cache(int64_t seqlen) {
    if (seqlen <= seq_len_cached_) return;

    seqlen *= 2;
    seq_len_cached_ = seqlen;

    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(inv_freq_.device());
    inv_freq_ =
        1.0 / torch::pow(theta_, torch::arange(0, dim_, 2, options) / dim_);
    auto seq = torch::arange(seqlen, options);
    freqs_cached_ = torch::outer(seq, inv_freq_);
  }

  torch::Tensor forward(int seqlen) {
    update_freqs_cache(seqlen);
    return freqs_cached_.slice(0, 0, seqlen);
  }

 private:
  int dim_ = 0;
  double theta_ = 0.0;

  int64_t seq_len_cached_ = 0;
  torch::Tensor inv_freq_;
  torch::Tensor freqs_cached_;
};
TORCH_MODULE(Qwen2_5_VisionRotaryEmbedding);

class Qwen2_5_VisionPatchMergerImpl : public torch::nn::Module {
 public:
  Qwen2_5_VisionPatchMergerImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    auto quant_args = context.get_quant_args();
    auto parallel_args = context.get_parallel_args();

    int64_t d_model = model_args.mm_projection_dim();  // out_hidden_size
    int context_dim = model_args.mm_hidden_size();
    int spatial_merge_size = model_args.mm_spatial_merge_size();

    hidden_size_ =
        context_dim * static_cast<int>(std::pow(spatial_merge_size, 2));
    ln_q_ = register_module("ln_q", RmsNorm(context));

    auto cpl = torch::nn::Linear(
        torch::nn::LinearOptions(hidden_size_, hidden_size_).bias(true));
    cpl->weight.set_data(cpl->weight.to(options));
    cpl->bias.set_data(cpl->bias.to(options));
    auto act = torch::nn::GELU();
    auto rpl = torch::nn::Linear(
        torch::nn::LinearOptions(hidden_size_, d_model).bias(true));
    rpl->weight.set_data(rpl->weight.to(options));
    rpl->bias.set_data(rpl->bias.to(options));
    mlp_ = register_module("mlp", torch::nn::Sequential(cpl, act, rpl));
    layers_ = std::make_tuple(cpl, act, rpl);
  }

  torch::Tensor forward(torch::Tensor x) {
    x = ln_q_(x, 0);
    x = x.view({-1, hidden_size_});
    return mlp_->forward(x);
  }

  void load_state_dict(const StateDict& state_dict) {
    ln_q_->load_state_dict(state_dict.get_dict_with_prefix("ln_q."));

    const auto& cpl_dict = state_dict.get_dict_with_prefix("mlp.0.");
    const auto& cpl_weight = cpl_dict.get_tensor("weight");
    if (cpl_weight.defined()) {
      CHECK_EQ(std::get<0>(layers_)->weight.sizes(), cpl_weight.sizes())
          << "weight size mismatch for " << name();
      std::get<0>(layers_)->weight.data().copy_(cpl_weight);
      is_cpl_weight_loaded = true;
    }
    const auto cpl_bias = cpl_dict.get_tensor("bias");
    if (cpl_bias.defined()) {
      CHECK_EQ(std::get<0>(layers_)->bias.sizes(), cpl_bias.sizes())
          << "bias size mismatch for " << name();
      std::get<0>(layers_)->bias.data().copy_(cpl_bias);
      is_cpl_bias_loaded = true;
    }

    const auto& rpl_dict = state_dict.get_dict_with_prefix("mlp.2.");
    const auto& rpl_weight = rpl_dict.get_tensor("weight");
    if (rpl_weight.defined()) {
      CHECK_EQ(std::get<2>(layers_)->weight.sizes(), rpl_weight.sizes())
          << "weight size mismatch for " << name();
      std::get<2>(layers_)->weight.data().copy_(rpl_weight);
      is_rpl_weight_loaded = true;
    }
    const auto rpl_bias = rpl_dict.get_tensor("bias");
    if (rpl_bias.defined()) {
      CHECK_EQ(std::get<2>(layers_)->bias.sizes(), rpl_bias.sizes())
          << "bias size mismatch for " << name();
      std::get<2>(layers_)->bias.data().copy_(rpl_bias);
      is_rpl_bias_loaded = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    ln_q_->verify_loaded_weights(prefix + "ln_q.");
    CHECK(is_cpl_weight_loaded)
        << "weight is not loaded for " << prefix + "mlp.0" + ".weight";
    CHECK(is_cpl_bias_loaded)
        << "bias is not loaded for " << prefix + "mlp.0" + ".bias";
    CHECK(is_rpl_weight_loaded)
        << "weight is not loaded for " << prefix + "mlp.2" + ".weight";
    CHECK(is_rpl_bias_loaded)
        << "bias is not loaded for " << prefix + "mlp.2" + ".bias";
  }

  void merge_loaded_weights() { ln_q_->merge_loaded_weights(); }

 private:
  int64_t hidden_size_;

  RmsNorm ln_q_{nullptr};
  torch::nn::Sequential mlp_{nullptr};
  std::tuple<torch::nn::Linear, torch::nn::GELU, torch::nn::Linear> layers_ = {
      nullptr,
      nullptr,
      nullptr};
  bool is_cpl_weight_loaded = false;
  bool is_cpl_bias_loaded = false;
  bool is_rpl_weight_loaded = false;
  bool is_rpl_bias_loaded = false;
};
TORCH_MODULE(Qwen2_5_VisionPatchMerger);

class Qwen2_5_VisionTransformerImpl : public torch::nn::Module {
 public:
  Qwen2_5_VisionTransformerImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();

    hidden_size_ = model_args.mm_hidden_size();
    num_heads_ = model_args.mm_num_attention_heads();

    window_size_ = model_args.mm_window_size();
    patch_size_ = model_args.mm_patch_size();
    spatial_merge_size_ = model_args.mm_spatial_merge_size();
    const auto& block_indexes = model_args.mm_fullatt_block_indexes();
    fullatt_block_indexes_.insert(block_indexes.begin(), block_indexes.end());
    spatial_merge_unit_ = static_cast<int>(std::pow(spatial_merge_size_, 2));

    patch_embed_ =
        register_module("patch_embed", Qwen2_5_VisionPatchEmbed(context));
    rotary_pos_emb_ = register_module("rotary_pos_emb",
                                      Qwen2_5_VisionRotaryEmbedding(context));
    blocks_ = register_module("blocks", torch::nn::ModuleList());

    for (int32_t idx = 0; idx < model_args.mm_num_hidden_layers(); idx++) {
      auto block = Qwen2_5_VisionBlock(context);
      blocks_->push_back(block);
      layers_.push_back(block);
    }
    merger_ = register_module("merger", Qwen2_5_VisionPatchMerger(context));
  }

  torch::Tensor rot_pos_emb(torch::Tensor grid_thw) {
    std::vector<torch::Tensor> pos_ids_vec;
    auto count = grid_thw.sizes()[0];
    pos_ids_vec.reserve(count);

    auto grid_thw_cpu = grid_thw.cpu();
    auto options =
        torch::TensorOptions().dtype(torch::kLong).device(grid_thw.device());

    for (int idx = 0; idx < count; ++idx) {
      auto t = grid_thw_cpu[idx][0].item<int64_t>();
      auto h = grid_thw_cpu[idx][1].item<int64_t>();
      auto w = grid_thw_cpu[idx][2].item<int64_t>();

      auto hpos_ids = torch::arange(h, options).unsqueeze(1).expand({-1, w});
      hpos_ids = hpos_ids
                     .reshape({h / spatial_merge_size_,
                               spatial_merge_size_,
                               w / spatial_merge_size_,
                               spatial_merge_size_})
                     .permute({0, 2, 1, 3})
                     .flatten();

      auto wpos_ids = torch::arange(w, options).unsqueeze(0).expand({h, -1});
      wpos_ids = wpos_ids
                     .reshape({h / spatial_merge_size_,
                               spatial_merge_size_,
                               w / spatial_merge_size_,
                               spatial_merge_size_})
                     .permute({0, 2, 1, 3})
                     .flatten();

      pos_ids_vec.push_back(
          torch::stack({hpos_ids, wpos_ids}, -1).repeat({t, 1}));
    }

    auto pos_ids = torch::cat(pos_ids_vec, 0);
    auto max_grid_size =
        grid_thw
            .index({torch::indexing::Slice(),
                    torch::indexing::Slice(1, torch::indexing::None)})
            .max();

    auto rotary_pos_emb_full = rotary_pos_emb_(max_grid_size.item<int64_t>());
    auto rotary_pos_emb = rotary_pos_emb_full.index({pos_ids}).flatten(1);

    return rotary_pos_emb;
  }

  torch::Tensor get_window_index(torch::Tensor grid_thw,
                                 std::vector<int>& cu_window_seqlens) {
    auto count = grid_thw.sizes()[0];
    std::vector<torch::Tensor> window_index;
    window_index.reserve(count);
    cu_window_seqlens.reserve(count * 128);
    cu_window_seqlens.emplace_back(0);

    int window_index_id = 0;
    int vit_merger_window_size =
        window_size_ / spatial_merge_size_ / patch_size_;

    auto grid_thw_cpu = grid_thw.cpu();
    auto options =
        torch::TensorOptions().dtype(torch::kLong).device(grid_thw.device());

    for (int idx = 0; idx < count; ++idx) {
      auto grid_t = grid_thw_cpu[idx][0].item<int64_t>();
      auto grid_h = grid_thw_cpu[idx][1].item<int64_t>();
      auto grid_w = grid_thw_cpu[idx][2].item<int64_t>();

      auto llm_grid_h = grid_h / spatial_merge_size_;
      auto llm_grid_w = grid_w / spatial_merge_size_;

      auto index = torch::arange(grid_t * llm_grid_h * llm_grid_w, options)
                       .reshape({grid_t, llm_grid_h, llm_grid_w});
      auto pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size;
      auto pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size;

      auto num_windows_h = (llm_grid_h + pad_h) / vit_merger_window_size;
      auto num_windows_w = (llm_grid_w + pad_w) / vit_merger_window_size;

      namespace F = torch::nn::functional;
      auto index_padded = F::pad(index,
                                 F::PadFuncOptions({0, pad_w, 0, pad_h})
                                     .mode(torch::kConstant)
                                     .value(-100));
      index_padded = index_padded.reshape({grid_t,
                                           num_windows_h,
                                           vit_merger_window_size,
                                           num_windows_w,
                                           vit_merger_window_size});

      index_padded = index_padded.permute({0, 1, 3, 2, 4})
                         .reshape({grid_t,
                                   num_windows_h * num_windows_w,
                                   vit_merger_window_size,
                                   vit_merger_window_size});

      auto index_padded_ne = torch::ne(index_padded, -100);
      auto seqlens = index_padded_ne.sum({2, 3}).reshape({-1});
      index_padded = index_padded.reshape({-1});
      auto index_new =
          index_padded.masked_select(index_padded_ne.reshape({-1}));

      window_index.push_back(index_new + window_index_id);
      auto cu_seqlens_tmp =
          (seqlens.cumsum(0, torch::kInt32) * spatial_merge_unit_ +
           cu_window_seqlens.back())
              .cpu();
      cu_window_seqlens.insert(
          cu_window_seqlens.end(),
          cu_seqlens_tmp.data_ptr<int>(),
          cu_seqlens_tmp.data_ptr<int>() + cu_seqlens_tmp.numel());
      window_index_id += grid_t * llm_grid_h * llm_grid_w;
    }

    return torch::cat(window_index, 0);
  }

  torch::Tensor forward(torch::Tensor hidden_states,
                        torch::Tensor grid_thw,  // [batch,thw]
                        const ModelInputParams& input_params) {
    // patchify
    // hidden_states = x.to(device=self.device, dtype=self.dtype);
    hidden_states = patch_embed_(hidden_states);
    //  compute position embedding
    auto rotary_pos_emb = rot_pos_emb(grid_thw);

    // windows attention
    std::vector<int> cu_window_seqlens_vec;
    auto window_index = get_window_index(grid_thw, cu_window_seqlens_vec);
    torch::TensorOptions options = torch::TensorOptions()
                                       .dtype(torch::kInt32)
                                       .device(hidden_states.device());
    auto cu_window_seqlens = torch::tensor(cu_window_seqlens_vec, options);
    cu_window_seqlens =
        std::get<0>(torch::unique_consecutive(cu_window_seqlens));
    auto seq_len = hidden_states.sizes()[0];
    hidden_states = hidden_states.reshape(
        {seq_len / spatial_merge_unit_, spatial_merge_unit_, -1});
    hidden_states = hidden_states.index(
        {window_index, torch::indexing::Slice(), torch::indexing::Slice()});
    hidden_states = hidden_states.reshape({seq_len, -1});

    rotary_pos_emb = rotary_pos_emb.reshape(
        {seq_len / spatial_merge_unit_, spatial_merge_unit_, -1});
    rotary_pos_emb = rotary_pos_emb.index(
        {window_index, torch::indexing::Slice(), torch::indexing::Slice()});
    rotary_pos_emb = rotary_pos_emb.reshape({seq_len, -1});

    // compute cu_seqlens
    auto cu_seqlens = torch::repeat_interleave(
                          grid_thw.index({torch::indexing::Slice(), 1}) *
                              grid_thw.index({torch::indexing::Slice(), 2}),
                          grid_thw.index({torch::indexing::Slice(), 0}))
                          .cumsum(0, torch::kInt32);
    namespace F = torch::nn::functional;
    cu_seqlens = F::pad(
        cu_seqlens, F::PadFuncOptions({1, 0}).mode(torch::kConstant).value(0));

    // transformers
    cu_seqlens = torch::diff(cu_seqlens);
    cu_window_seqlens = torch::diff(cu_window_seqlens);
    m_cos = rotary_pos_emb.cos().type_as(hidden_states);
    m_cos = torch::nn::functional::pad(
        m_cos, torch::nn::functional::PadFuncOptions({0, 24}));
    m_cos = m_cos.repeat({1, 2});
    m_sin = rotary_pos_emb.sin().type_as(hidden_states);
    m_sin = torch::nn::functional::pad(
        m_sin, torch::nn::functional::PadFuncOptions({0, 24}));
    m_sin = m_sin.repeat({1, 2});
    ModelInputParams& input_params_new =
        const_cast<ModelInputParams&>(input_params);
    torch::Tensor cu_seqlens_cpu = cu_seqlens.cpu();
    torch::Tensor cu_window_seqlens_cpu = cu_window_seqlens.cpu();
    std::vector<int> cu_seqlens_vec(
        cu_seqlens_cpu.data_ptr<int>(),  // full seqlen vec
        cu_seqlens_cpu.data_ptr<int>() + cu_seqlens_cpu.numel());
    std::vector<int> cu_w_seqlens_vec(
        cu_window_seqlens_cpu.data_ptr<int>(),  // windows seqlen vec
        cu_window_seqlens_cpu.data_ptr<int>() + cu_window_seqlens_cpu.numel());
    for (int idx = 0; idx < blocks_->size(); ++idx) {
      torch::Tensor cu_seqlens_now;
      std::vector<int> cu_seqlens_now_vec;
      if (fullatt_block_indexes_.find(idx) != fullatt_block_indexes_.end()) {
        cu_seqlens_now = cu_seqlens;
        cu_seqlens_now_vec = cu_seqlens_vec;
      } else {
        cu_seqlens_now = cu_window_seqlens;
        cu_seqlens_now_vec = cu_w_seqlens_vec;
      }
      hidden_states = layers_[idx](hidden_states,
                                   m_cos,
                                   m_sin,
                                   cu_seqlens_now,
                                   cu_seqlens_now_vec,
                                   input_params_new,
                                   idx);
    }
    // adapter
    hidden_states = merger_(hidden_states);

    auto reverse_indices = torch::argsort(window_index);
    hidden_states =
        hidden_states.index({reverse_indices, torch::indexing::Slice()});
    return hidden_states;
  }

  void load_state_dict(const StateDict& state_dict) {
    patch_embed_->load_state_dict(
        state_dict.get_dict_with_prefix("patch_embed."));
    for (int idx = 0; idx < blocks_->size(); ++idx) {
      layers_[idx]->load_state_dict(state_dict.get_dict_with_prefix(
          "blocks." + std::to_string(idx) + "."));
    }

    merger_->load_state_dict(state_dict.get_dict_with_prefix("merger."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    patch_embed_->verify_loaded_weights(prefix + "patch_embed.");
    for (int idx = 0; idx < blocks_->size(); ++idx) {
      layers_[idx]->verify_loaded_weights(prefix + "blocks." +
                                          std::to_string(idx) + ".");
    }
    merger_->verify_loaded_weights(prefix + "merger.");
  }

  void merge_loaded_weights() {
    for (int idx = 0; idx < blocks_->size(); ++idx) {
      layers_[idx]->merge_loaded_weights();
    }
    merger_->merge_loaded_weights();
  }

 private:
  int hidden_size_ = 0;
  int num_heads_ = 0;
  int window_size_ = 0;
  int patch_size_ = 0;
  int spatial_merge_size_ = 0;
  std::set<int> fullatt_block_indexes_;
  int spatial_merge_unit_ = 0;

  Qwen2_5_VisionPatchEmbed patch_embed_{nullptr};
  Qwen2_5_VisionRotaryEmbedding rotary_pos_emb_{nullptr};
  torch::nn::ModuleList blocks_{nullptr};
  std::vector<Qwen2_5_VisionBlock> layers_;
  Qwen2_5_VisionPatchMerger merger_{nullptr};

  torch::Tensor m_cos;
  torch::Tensor m_sin;
  int device_id = 0;
};
TORCH_MODULE(Qwen2_5_VisionTransformer);

struct Qwen2_5_VLImageInputs {
  torch::Tensor pixel_values;
  torch::Tensor image_grid_thw;
};

struct Qwen2_5_VLVideoInputs {
  torch::Tensor pixel_values_videos;
  torch::Tensor video_grid_thw;
  torch::Tensor second_per_grid_ts;
};

class Qwen2_5_VLForConditionalGenerationImpl : public torch::nn::Module {
 public:
  Qwen2_5_VLForConditionalGenerationImpl(const ModelContext& context)
      : model_args_(context.get_model_args()),
        options_(context.get_tensor_options()) {
    visual_ = register_module("visual", Qwen2_5_VisionTransformer(context));

    language_model_ =
        register_module("language_model", QWen2ForCausalLM(context));
  }

  torch::Tensor get_input_embeddings(
      torch::Tensor input_ids,
      const std::optional<Qwen2_5_VLImageInputs>& image_input,
      const std::optional<Qwen2_5_VLVideoInputs>& video_input,
      const ModelInputParams& input_params) {
    auto inputs_embeds = language_model_->get_input_embeddings(input_ids);
    if (image_input) {
      // visual
      auto image_embeds = visual_(image_input->pixel_values.to(options_),
                                  image_input->image_grid_thw,
                                  input_params);
      // merge
      auto is_multimodal = torch::isin(input_ids, model_args_.image_token_id());
      inputs_embeds.index_put_({is_multimodal}, image_embeds);
    }
    if (video_input) {
      // visual
      auto video_embeds = visual_(video_input->pixel_values_videos.to(options_),
                                  video_input->video_grid_thw,
                                  input_params);
      // merge
      auto is_multimodal = torch::isin(input_ids, model_args_.video_token_id());
      inputs_embeds.index_put_({is_multimodal}, video_embeds);
    }
    return inputs_embeds;
  }

  torch::Tensor forward(const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches,
                        const ModelInputParams& input_params) {
    torch::NoGradGuard no_grad;
    const auto& mm_data = input_params.mm_data;

    torch::Tensor pixel_values;
    if (const auto& res = mm_data.get<torch::Tensor>("pixel_values"))
      pixel_values = res.value();

    torch::Tensor image_grid_thw;
    if (const auto& res = mm_data.get<torch::Tensor>("image_grid_thw"))
      image_grid_thw = res.value();

    torch::Tensor pixel_values_videos;
    if (const auto& res = mm_data.get<torch::Tensor>("pixel_values_videos"))
      pixel_values_videos = res.value();

    torch::Tensor video_grid_thw;
    if (const auto& res = mm_data.get<torch::Tensor>("video_grid_thw"))
      video_grid_thw = res.value();

    torch::Tensor second_per_grid_ts;
    if (const auto& res = mm_data.get<torch::Tensor>("second_per_grid_ts"))
      second_per_grid_ts = res.value();

    std::optional<Qwen2_5_VLImageInputs> image_inputs;
    std::optional<Qwen2_5_VLVideoInputs> video_inputs;

    if (pixel_values.defined() && image_grid_thw.defined())
      image_inputs = Qwen2_5_VLImageInputs{pixel_values, image_grid_thw};

    if (pixel_values_videos.defined() && video_grid_thw.defined() &&
        second_per_grid_ts.defined())
      video_inputs = Qwen2_5_VLVideoInputs{
          pixel_values_videos, video_grid_thw, second_per_grid_ts};

    auto inputs_embeds =
        get_input_embeddings(tokens, image_inputs, video_inputs, input_params);
    input_params.input_embedding = inputs_embeds;

    auto emb = language_model_(tokens, positions, kv_caches, input_params);

    return emb;
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    return language_model_->logits(hidden_states, seleted_idxes);
  }

  void load_model(std::unique_ptr<ModelLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      visual_->load_state_dict(state_dict->get_dict_with_prefix("visual."));
    }
    // verify
    visual_->verify_loaded_weights("visual.");
    visual_->merge_loaded_weights();
    if (!model_args_.image_embedding_mode()) {
      language_model_->load_model(std::move(loader));
    }
  }

  LlmHead get_lm_head() { return language_model_->get_lm_head(); }
  void set_lm_head(LlmHead& head) { language_model_->set_lm_head(head); }

  AtbWordEmbedding get_word_embedding() {
    return language_model_->get_word_embedding();
  }

  void set_word_embedding(AtbWordEmbedding& word_embedding) {
    language_model_->set_word_embedding(word_embedding);
  }

 private:
  ModelArgs model_args_;
  torch::TensorOptions options_;

  Qwen2_5_VisionTransformer visual_{nullptr};
  QWen2ForCausalLM language_model_{nullptr};
};
TORCH_MODULE(Qwen2_5_VLForConditionalGeneration);

REGISTER_INPUT_PROCESSOR(qwen2_5_vl, Qwen2_5_VLInputProcessor);
REGISTER_CAUSAL_VLM_MODEL(qwen2_5_vl, Qwen2_5_VLForConditionalGeneration);
REGISTER_IMAGE_PROCESSOR(qwen2_5_vl, Qwen2VLImageProcessor);

REGISTER_MODEL_ARGS(qwen2_5_vl, [&] {
  // text config
  // LOAD_ARG_OR(attention_dropout, "attention_dropout", 0.0);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 151643);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 151645);
  LOAD_ARG_OR(vision_start_token_id, "vision_start_token_id", 151652);
  LOAD_ARG_OR(vision_end_token_id, "vision_end_token_id", 151653);
  LOAD_ARG_OR(vision_token_id, "vision_token_id", 151654);
  LOAD_ARG_OR(image_token_id, "image_token_id", 151655);
  LOAD_ARG_OR(video_token_id, "video_token_id", 151656);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(hidden_size, "hidden_size", 3584);
  // LOAD_ARG_OR(initializer_range, "initializer_range", 0.02);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 18944);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 128000);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 28);
  LOAD_ARG_OR(model_type, "model_type", "qwen2_5_vl");
  LOAD_ARG_OR(n_heads, "num_attention_heads", 28);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 28);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 4);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-06);
  LOAD_ARG_OR(rope_theta, "rope_theta", 1000000.0f);
  LOAD_ARG_OR(sliding_window, "sliding_window", 32768);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  // LOAD_ARG_OR(transformers_version, "transformers_version", "4.41.2");
  // LOAD_ARG_OR(use_cache, "use_cache", true);
  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });

  // vision_config
  LOAD_ARG_OR(mm_num_hidden_layers, "vision_config.depth", 32);
  LOAD_ARG_OR(mm_hidden_act, "vision_config.hidden_act", "silu");
  LOAD_ARG_OR(mm_hidden_size, "vision_config.hidden_size", 1280);
  LOAD_ARG_OR(mm_intermediate_size, "vision_config.intermediate_size", 3420);
  LOAD_ARG_OR(mm_num_attention_heads, "vision_config.num_heads", 16);
  LOAD_ARG_OR(mm_num_channels, "vision_config.in_chans", 3);
  LOAD_ARG_OR(mm_projection_dim, "vision_config.out_hidden_size", 3584);
  LOAD_ARG_OR(mm_patch_size, "vision_config.patch_size", 14);
  LOAD_ARG_OR(mm_spatial_merge_size, "vision_config.spatial_merge_size", 2);
  LOAD_ARG_OR(mm_spatial_patch_size, "vision_config.spatial_patch_size", 14);
  LOAD_ARG_OR(mm_window_size, "vision_config.window_size", 112);
  LOAD_ARG(mm_fullatt_block_indexes, "vision_config.fullatt_block_indexes");
  LOAD_ARG_OR(mm_tokens_per_second, "vision_config.tokens_per_second", 2);
  LOAD_ARG_OR(mm_temporal_patch_size, "vision_config.temporal_patch_size", 2);
  LOAD_ARG_OR_FUNC(mm_head_dim, "head_dim", [&] {
    return args->mm_hidden_size() / args->mm_num_attention_heads();
  });

  LOAD_ARG_OR(
      rope_scaling_rope_type, "vision_config.rope_scaling.type", "mrope");
  LOAD_ARG(rope_scaling_mrope_section, "rope_scaling.mrope_section");
  LOAD_ARG_OR(vocab_size, "vocab_size", 152064);
});
}  // namespace xllm::hf
