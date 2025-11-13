#pragma once

#include "core/framework/model/embedding_lm.h"
#include "models/llm/qwen2.h"
#include "models/vlm/qwen2_vl.h"
namespace xllm {

class Qwen2_VLForEmbeddingImpl : public torch::nn::Module {
 public:
  Qwen2_VLForEmbeddingImpl(const ModelContext& context)
      : model_args_(context.get_model_args()),
        options_(context.get_tensor_options()) {
    visual_ = register_module("visual", Qwen2_VisionTransformer(context));
    language_model_ =
        register_module("language_model", QWen2ForCausalLM(context));
  }

  torch::Tensor get_input_embeddings(
      torch::Tensor input_ids,
      const std::optional<Qwen2_VLImageInputs>& image_input,
      const std::optional<Qwen2_VLVideoInputs>& video_input,
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
    return inputs_embeds;
  }

  torch::Tensor forward(const std::vector<torch::Tensor>& tokens,
                        const std::vector<torch::Tensor>& positions,
                        std::vector<KVCache>& kv_caches,
                        const std::vector<ModelInputParams>& input_params) {
    torch::NoGradGuard no_grad;
    const auto& mm_data = input_params[0].mm_data;
    torch::Tensor pixel_values;
    if (const auto& res = mm_data.get<torch::Tensor>("pixel_values"))
      pixel_values = res.value();

    torch::Tensor image_grid_thw;
    if (const auto& res = mm_data.get<torch::Tensor>("image_grid_thw"))
      image_grid_thw = res.value();
    std::optional<Qwen2_VLImageInputs> image_inputs;
    std::optional<Qwen2_VLVideoInputs> video_inputs;

    if (pixel_values.defined() && image_grid_thw.defined())
      image_inputs = Qwen2_VLImageInputs{pixel_values, image_grid_thw};
    auto inputs_embeds = get_input_embeddings(
        tokens[0], image_inputs, video_inputs, input_params[0]);
    input_params[0].input_embedding = inputs_embeds;
    auto emb = language_model_(tokens, positions, kv_caches, input_params);
    auto emb_out = pooler(emb, torch::Tensor());
    return emb_out;
  }

  torch::Tensor pooler(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    auto h = hidden_states;
    if (seleted_idxes.defined()) {
      h = h.index_select(/*dim=*/0, seleted_idxes);
    }
    auto pooler_output = torch::nn::functional::normalize(
        h, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
    return pooler_output;
  }

  torch::Tensor logits(const torch::Tensor&, const torch::Tensor&) {
    LOG(ERROR) << "logits() not implemented for Embedding Model!";
    return torch::empty({0});
  }

  torch::Device device() const { return options_.device(); }

  const torch::TensorOptions& options() const { return options_; }

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

  layer::LmHead get_lm_head() { return language_model_->get_lm_head(); }
  void set_lm_head(layer::LmHead& head) { language_model_->set_lm_head(head); }

  std::vector<layer::WordEmbedding> get_word_embedding() {
    return language_model_->get_word_embedding();
  }

  void set_word_embedding(std::vector<layer::WordEmbedding>& word_embedding) {
    language_model_->set_word_embedding(word_embedding);
  }

 private:
  ModelArgs model_args_;
  torch::TensorOptions options_;

  Qwen2_VisionTransformer visual_{nullptr};
  QWen2ForCausalLM language_model_{nullptr};
};
TORCH_MODULE(Qwen2_VLForEmbedding);

REGISTER_INPUT_PROCESSOR(qwen2_vl_embedding, Qwen2_5_VLInputProcessor);
REGISTER_CAUSAL_VLM_MODEL(qwen2_vl_embedding, Qwen2_VLForEmbedding);
REGISTER_IMAGE_PROCESSOR(qwen2_vl_embedding, Qwen2VLImageProcessor);

REGISTER_MODEL_ARGS(qwen2_vl_embedding, [&] {
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
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 32768);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 28);
  LOAD_ARG_OR(model_type, "model_type", "qwen2_vl_embedding");
  LOAD_ARG_OR(n_heads, "num_attention_heads", 28);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 28);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 4);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-06);
  LOAD_ARG_OR(rope_theta, "rope_theta", 1000000.0f);
  LOAD_ARG_OR(sliding_window, "sliding_window", 32768);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);
  LOAD_ARG_OR(dtype, "torch_dtype", "bfloat16");
  // LOAD_ARG_OR(transformers_version, "transformers_version", "4.41.2");
  // LOAD_ARG_OR(use_cache, "use_cache", true);
  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });

  // vision_config
  LOAD_ARG_OR(mm_num_hidden_layers, "vision_config.depth", 32);
  // LOAD_ARG_OR(mm_hidden_act, "vision_config.hidden_act", "silu");
  LOAD_ARG_OR(mm_hidden_size, "vision_config.embed_dim", 1280);
  // LOAD_ARG_OR(mm_mlp_ratio, "vision_config.mlp_ratio", 4);
  LOAD_ARG_OR(mm_num_attention_heads, "vision_config.num_heads", 16);
  LOAD_ARG_OR(mm_num_channels, "vision_config.in_chans", 3);
  LOAD_ARG_OR(mm_projection_dim, "vision_config.hidden_size", 3584);
  LOAD_ARG_OR(mm_patch_size, "vision_config.patch_size", 14);
  LOAD_ARG_OR(mm_spatial_merge_size, "vision_config.spatial_merge_size", 2);
  LOAD_ARG_OR(mm_spatial_patch_size, "vision_config.spatial_patch_size", 14);
  LOAD_ARG_OR(mm_temporal_patch_size, "vision_config.temporal_patch_size", 2);
  LOAD_ARG_OR_FUNC(mm_head_dim, "head_dim", [&] {
    return args->mm_hidden_size() / args->mm_num_attention_heads();
  });

  LOAD_ARG_OR(rope_scaling_rope_type, "rope_scaling.type", "mrope");
  LOAD_ARG(rope_scaling_mrope_section, "rope_scaling.mrope_section");
  LOAD_ARG_OR(vocab_size, "vocab_size", 152064);
});
}  // namespace xllm
