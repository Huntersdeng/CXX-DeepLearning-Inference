#include "model/clip/text_encoder.h"

#include <yaml-cpp/yaml.h>

using namespace clip;

TextEncoder::TextEncoder(const std::string &yaml_file) : m_input_size_(77), m_output_size_(512) {
    YAML::Node yaml_node = YAML::LoadFile(yaml_file);

    std::string bpe_path = yaml_node["bpe_path"].as<std::string>();
    m_tokenizer_ = std::make_shared<TextTokenizer>(bpe_path);

    std::string model_path = yaml_node["model_path"].as<std::string>();
    std::string framework_type = yaml_node["framework"].as<std::string>();

    if (!Init(model_path, framework_type)) exit(0);

    config_.input_len["TEXT"] = 2 * m_input_size_;
    config_.output_len["TEXT_EMBEDDING"] = 2 * m_output_size_;
    config_.is_dynamic = true;
    Status status = framework_->Init(config_);
    if (status != Status::SUCCESS) {
        std::cout << "Failed to init framework" << std::endl;
        exit(0);
    }
}

TextEncoder::~TextEncoder() { std::cout << "Destruct text encoder" << std::endl; }

void TextEncoder::preprocess(const std::vector<std::string> &texts, IOTensor &text_embeddings) {
    std::vector<std::vector<int>> tokens = m_tokenizer_->batchTokenize(texts);
    std::vector<int> tensor;
    for (const auto &token : tokens) {
        for (int i : token) {
            tensor.push_back(i);
        }
    }

    text_embeddings.resize(tensor.size() * sizeof(int));
    text_embeddings.shape =
        std::vector<int64_t>{static_cast<int64_t>(texts.size()), static_cast<int64_t>(tokens[0].size())};
    text_embeddings.data_type = DataType::INT32;
    memcpy(text_embeddings.data(), tensor.data(), text_embeddings.size());
}

void TextEncoder::forward(const std::vector<std::string> &texts, IOTensor &features) {
    std::unordered_map<std::string, IOTensor> input, output;

    input["TEXT"] = IOTensor();
    preprocess(texts, input["TEXT"]);

    output["TEXT_EMBEDDING"] = IOTensor();
    output["TEXT_EMBEDDING"].resize(texts.size() * m_output_size_ * sizeof(float));
    output["TEXT_EMBEDDING"].shape =
        std::vector<int64_t>{static_cast<int64_t>(texts.size()), static_cast<int64_t>(m_output_size_)};
    output["TEXT_EMBEDDING"].data_type = DataType::FP32;

    this->framework_->forward(input, output);

    features.resize(output["TEXT_EMBEDDING"].size());
    memcpy(features.data(), output["TEXT_EMBEDDING"].data(), features.size());
    features.shape = output["TEXT_EMBEDDING"].shape;
    features.data_type = output["TEXT_EMBEDDING"].data_type;
}