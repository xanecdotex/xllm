// qwen2_vl_embedding.h
#pragma once
#include "core/framework/model/embedding_lm.h"
#include "models/llm/qwen2.h"
#include "models/vlm/qwen2_vl.h"
namespace xllm {

struct Qwen2_VLImageInputs {
  torch::Tensor pixel_values;
  torch::Tensor image_grid_thw;
};

struct Qwen2_VLVideoInputs {
  torch::Tensor pixel_values_videos;
  torch::Tensor video_grid_thw;
  torch::Tensor second_per_grid_ts;
};

class Qwen2_VLForEmbeddingImpl : public torch::nn::Module {
 public:
  Qwen2_VLForEmbeddingImpl(const ModelContext& ctx)
      : args_(ctx.get_model_args()), options_(ctx.get_tensor_options()) {
    visual_ = register_module("visual", Qwen2_VisionTransformer(ctx));
    language_model = register_module("language_model", QWen2ForCausalLM(ctx));
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

    return emb;
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

  layer::LmHead get_language_modelhead() {
    return language_model_->get_language_modelhead();
  }

  void set_language_modelhead(layer::LmHead& head) {
    language_model_->set_language_modelhead(head);
  }

  std::vector<layer::WordEmbedding> get_word_embedding() {
    return language_model_->get_word_embedding();
  }

  void set_word_embedding(std::vector<layer::WordEmbedding>& word_embedding) {
    language_model_->set_word_embedding(word_embedding);
  }

 private:
  ModelArgs args_;
  torch::TensorOptions options_;

  Qwen2_VisionTransformer visual_{nullptr};
  QWen2ForCausalLM language_model{nullptr};
};
TORCH_MODULE(Qwen2_VLForEmbedding);

REGISTER_EMBEDDING_MODEL_WITH_VARNAME(qwen2_vl_embedding,
                                      qwen2_vl,
                                      Qwen2_VLForEmbedding);

}  // namespace xllm
