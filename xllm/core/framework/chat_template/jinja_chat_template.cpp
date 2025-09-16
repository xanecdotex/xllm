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

#include "jinja_chat_template.h"

#include <glog/logging.h>
#include <unistd.h>

#include <optional>
#include <string>

namespace xllm {

JinjaChatTemplate::JinjaChatTemplate(const TokenizerArgs& args) : args_(args) {
  try {
    template_ = std::make_unique<minja::chat_template>(
        args_.chat_template(), args_.bos_token(), args_.eos_token());
    LOG(INFO) << "Jinja chat template init succeed.";

  } catch (const std::exception& e) {
    LOG(FATAL) << "Failed to parse jinja chat template, TokenizerArgs: "
               << args_ << std::endl
               << "Error message: " << e.what();
  }
}

std::optional<std::string> JinjaChatTemplate::apply(
    const ChatMessages& messages) const {
  const std::vector<xllm::JsonTool> empty_tools;
  const nlohmann::ordered_json chat_template_kwargs;
  return apply(messages, empty_tools, chat_template_kwargs);
}

std::optional<std::string> JinjaChatTemplate::apply(
    const ChatMessages& messages,
    const nlohmann::ordered_json& chat_template_kwargs) const {
  const std::vector<xllm::JsonTool> empty_tools;
  return apply(messages, empty_tools, chat_template_kwargs);
}

std::optional<std::string> JinjaChatTemplate::apply(
    nlohmann::ordered_json& messages) const {
  // Call the overloaded method with empty tools
  nlohmann::ordered_json empty_tools = nlohmann::json::array();
  const nlohmann::ordered_json chat_template_kwargs = nlohmann::json::object();
  return apply(messages, empty_tools, chat_template_kwargs);
}

std::optional<std::string> JinjaChatTemplate::apply(
    const ChatMessages& messages,
    const std::vector<xllm::JsonTool>& json_tools,
    const nlohmann::ordered_json& chat_template_kwargs) const {
  // convert the messages to json object
  nlohmann::ordered_json messages_json = nlohmann::json::array();
  for (const auto& message : messages) {
    nlohmann::ordered_json message_json;
    message_json["role"] = message.role;

    if (std::holds_alternative<std::string>(message.content)) {
      message_json["content"] = std::get<std::string>(message.content);
    } else if (std::holds_alternative<Message::MMContentVec>(message.content)) {
      message_json["content"] =
          get_mm_content(std::get<Message::MMContentVec>(message.content));
    }

    messages_json.push_back(message_json);
  }

  nlohmann::ordered_json tools_json = nlohmann::json::array();
  for (const auto& json_tool : json_tools) {
    nlohmann::ordered_json tool_json;
    tool_json["type"] = json_tool.type;

    nlohmann::ordered_json function_json;
    function_json["name"] = json_tool.function.name;
    function_json["description"] = json_tool.function.description;
    function_json["parameters"] = json_tool.function.parameters;

    tool_json["function"] = function_json;
    tools_json.push_back(tool_json);
  }
  // apply the template
  return apply(messages_json, tools_json, chat_template_kwargs);
}

std::optional<std::string> JinjaChatTemplate::apply(
    nlohmann::ordered_json& messages,
    const nlohmann::ordered_json& tools,
    const nlohmann::ordered_json& chat_template_kwargs) const {
  for (auto& msg : messages) {
    if (!msg.contains("content")) continue;
    auto& content = msg["content"];
    auto normalize_item = [](nlohmann::ordered_json& item) {
      if (item.contains("type") && item["type"].is_string()) {
        std::string t = item["type"].get<std::string>();
        if (t == "video_url") item["type"] = "video";
      }
      if (item.contains("video_url") && !item.contains("video"))
        item["video"] = item["video_url"];
    };

    if (content.is_array()) {
      for (auto& it : content) normalize_item(it);
    } else if (content.is_object()) {
      normalize_item(content);
    }
  }
  minja::chat_template_inputs input;
  input.messages = messages;
  input.tools = tools;
  input.add_generation_prompt = true;
  input.extra_context = chat_template_kwargs;
  minja::chat_template_options options;

  return template_->apply(input, options);
}

nlohmann::ordered_json JinjaChatTemplate::get_mm_content(
    const Message::MMContentVec& vec) const {
  nlohmann::ordered_json content_json = nlohmann::json::array();

  for (const auto& item : vec) {
    nlohmann::ordered_json item_json;
    item_json["type"] = item.type;

    if (item.type == "text") {
      item_json["text"] = item.text;
    } else {
      item_json[item.type] = "mm place holder";
    }

    content_json.emplace_back(item_json);
  }

  return std::move(content_json);
}

}  // namespace xllm
