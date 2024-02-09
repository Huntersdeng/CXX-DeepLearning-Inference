#include "model/base/ocr_model.h"
#include <yaml-cpp/yaml.h>

OcrModel::OcrModel(const std::string &model_path, const std::string framework_type, cv::Size input_size, size_t input_channel, size_t output_size,
           const std::string alphabet)
    : m_input_size_(input_size), m_input_channel_(input_channel), m_output_length_(output_size) {
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

    alphabet_ = alphabet;

    config_.input_len["images"] = m_input_size_.height * m_input_size_.width * m_input_channel_;
    config_.output_len["output"] = m_output_length_;
    config_.is_dynamic = false;
    Status status = framework_->Init(config_);
    if (status != Status::SUCCESS) {
        std::cout << "Failed to init framework" << std::endl;
        exit(0);
    }
}

OcrModel::OcrModel(const std::string &yaml_file) {
    YAML::Node yaml_node = YAML::LoadFile(yaml_file);

    config_.model_path = yaml_node["model_path"].as<std::string>();

    std::string framework_type = yaml_node["framework"].as<std::string>();
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

    std::vector<long> input_size = yaml_node["input_size"].as<std::vector<long>>();
    m_input_size_.width = input_size.at(0);
    m_input_size_.height = input_size.at(1);
    m_input_channel_ = yaml_node["input_channel"].as<int>();

    m_output_length_ = yaml_node["output_size"].as<long>();

    alphabet_ = yaml_node["alphabet"].as<std::string>();

    config_.input_len["images"] = m_input_size_.height * m_input_size_.width * m_input_channel_;
    config_.output_len["output"] = m_output_length_;
    config_.is_dynamic = false;
    Status status = framework_->Init(config_);
    if (status != Status::SUCCESS) {
        std::cout << "Failed to init framework" << std::endl;
        exit(0);
    }
}

OcrModel::~OcrModel() { std::cout << "Destruct ocr model" << std::endl; }