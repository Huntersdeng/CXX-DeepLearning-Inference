#include "model/base/ocr_model.h"
#include <yaml-cpp/yaml.h>

OcrModel::OcrModel(const std::string &yaml_file) {
    YAML::Node yaml_node = YAML::LoadFile(yaml_file);

    std::string model_path = yaml_node["model_path"].as<std::string>();

    std::string framework_type = yaml_node["framework"].as<std::string>();
    if (!Init(model_path, framework_type)) exit(0);

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