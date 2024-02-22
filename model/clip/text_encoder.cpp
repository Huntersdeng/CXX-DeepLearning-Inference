#include "model/clip/text_encoder.h"

#include <yaml-cpp/yaml.h>

using namespace clip;

TextEncoder::TextEncoder(const std::string &model_path, const std::string framework_type, const std::string &bpe_path)
    : m_input_size_(77), m_output_size_(512) {
    m_tokenizer_ = std::make_shared<TextTokenizer>(bpe_path);
    config_.model_path = model_path;
    if (framework_type == "TensorRT") {
#ifdef USE_TENSORRT
        framework_ = std::make_shared<TensorRTFramework>();
#else
        std::cout << "Framework " << framework_type << " not implemented" << std::endl;
        exit(0);
#endif
    } else if (framework_type == "ONNX") {
        framework_ = std::make_shared<ONNXFramework>();
    } else {
        std::cout << "Framework " << framework_type << " not implemented" << std::endl;
        exit(0);
    }

    config_.input_len["TEXT"] = m_input_size_;
    config_.output_len["TEXT_EMBEDDING"] = m_output_size_;
    config_.is_dynamic = true;
    Status status = framework_->Init(config_);
    if (status != Status::SUCCESS) {
        std::cout << "Failed to init framework" << std::endl;
        exit(0);
    }
}

TextEncoder::TextEncoder(const std::string &yaml_file) : m_input_size_(77), m_output_size_(512) {
    YAML::Node yaml_node = YAML::LoadFile(yaml_file);

    bool online = yaml_node["online"].as<bool>();
    if (online) {
        std::string bpe_path = yaml_node["bpe_path"].as<std::string>();
        m_tokenizer_ = std::make_shared<TextTokenizer>(bpe_path);

        std::string model_path = yaml_node["model_path"].as<std::string>();
        std::string framework_type = yaml_node["framework"].as<std::string>();

        config_.model_path = model_path;
        if (framework_type == "TensorRT") {
    #ifdef USE_TENSORRT
            framework_ = std::make_shared<TensorRTFramework>();
    #else
            std::cout << "Framework " << framework_type << " not implemented" << std::endl;
            exit(0);
    #endif
        } else if (framework_type == "ONNX") {
            framework_ = std::make_shared<ONNXFramework>();
        } else {
            std::cout << "Framework " << framework_type << " not implemented" << std::endl;
            exit(0);
        }

        config_.input_len["TEXT"] = m_input_size_;
        config_.output_len["TEXT_EMBEDDING"] = m_output_size_;
        config_.is_dynamic = true;
        Status status = framework_->Init(config_);
        if (status != Status::SUCCESS) {
            std::cout << "Failed to init framework" << std::endl;
            exit(0);
        }
    } else {

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

void TextEncoder::setPrompt(const std::vector<std::string> &texts) {
    if (online) {
        std::unordered_map<std::string, IOTensor> input, output;

        input["TEXT"] = IOTensor();
        preprocess(texts, input["TEXT"]);

        output["TEXT_EMBEDDING"] = IOTensor();
        output["TEXT_EMBEDDING"].resize(texts.size() * config_.output_len["TEXT_EMBEDDING"] * sizeof(float));
        output["TEXT_EMBEDDING"].shape =
            std::vector<int64_t>{static_cast<int64_t>(texts.size()), config_.output_len["TEXT_EMBEDDING"]};
        output["TEXT_EMBEDDING"].data_type = DataType::FP32;

        this->framework_->forward(input, output);

        float *ptr = (float*)output["TEXT_EMBEDDING"].data();
        for (size_t i = 0; i < texts.size(); i++) {
            std::vector<float> embedding(m_output_size_, 0.0);
            for (size_t j = 0; j < m_output_size_; j++) {
                embedding[j] = *ptr++;
            }
            m_encoder_[texts[i]] = embedding;
        }
    } else {
        std::cout << "The text encoder is offline. Set prompt failed" << std::endl;
    }
    
}

void TextEncoder::forward(const std::vector<std::string> &texts, IOTensor &features) {
    features.resize(texts.size() * m_output_size_ * sizeof(float));
    features.shape = std::vector<int64_t>{static_cast<int64_t>(texts.size()), config_.output_len["TEXT_EMBEDDING"]};
    features.data_type = DataType::FP32;

    float *ptr = (float*)features.data();
    for (size_t i = 0; i < texts.size(); i++) {
        memcpy(ptr, m_encoder_[texts[i]].data(), m_output_size_ * sizeof(float));
        ptr += m_output_size_;
    }
}