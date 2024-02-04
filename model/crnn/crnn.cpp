#include "model/crnn/crnn.h"

#include <yaml-cpp/yaml.h>

Crnn::Crnn(const std::string &model_path, const std::string framework_type, cv::Size input_size, size_t output_size,
           const std::string alphabet)
    : m_input_size_(input_size), m_output_length_(output_size) {
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

    config_.input_len["images"] = m_input_size_.height * m_input_size_.width;
    config_.output_len["output"] = m_output_length_;
    Status status = framework_->Init(config_);
    if (status != Status::SUCCESS) {
        std::cout << "Failed to init framework" << std::endl;
        exit(0);
    }
}

Crnn::Crnn(const std::string &yaml_file) {
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

    m_output_length_ = yaml_node["output_size"].as<long>();

    alphabet_ = yaml_node["alphabet"].as<std::string>();

    config_.input_len["images"] = m_input_size_.height * m_input_size_.width;
    config_.output_len["output"] = m_output_length_;
    Status status = framework_->Init(config_);
    if (status != Status::SUCCESS) {
        std::cout << "Failed to init framework" << std::endl;
        exit(0);
    }
}

Crnn::~Crnn() { std::cout << "Destruct crnn" << std::endl; }

std::string Crnn::detect(const cv::Mat &image) {
    std::unordered_map<std::string, IOTensor> input, output;

    // 输入tensor设置
    cv::Mat nchw;
    cv::dnn::blobFromImage(image, nchw, 1 / 127.5f, m_input_size_, cv::Scalar(127.5, 127.5, 127.5), true, false, CV_32F);

    input["images"] = IOTensor();
    input["images"].resize(nchw.total() * nchw.elemSize());
    memcpy(input["images"].data(), nchw.ptr<uint8_t>(), nchw.total() * nchw.elemSize());

    // 输出张量设置
    output["output"] = IOTensor();
    output["output"].resize(config_.output_len["output"] * sizeof(int64_t));

    this->framework_->forward(input, output);
    return postprocess(output);
}

std::string Crnn::postprocess(const std::unordered_map<std::string, IOTensor> &output) {
    int64_t *const outputs = (int64_t *)output.at("output").data();
    std::string str;
    for (size_t i = 0; i < m_output_length_; i++) {
        if (outputs[i] == 0 || (i > 0 && outputs[i - 1] == outputs[i])) continue;
        str.push_back(alphabet_[outputs[i] - 1]);
    }
    return str;
}