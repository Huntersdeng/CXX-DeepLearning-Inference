#include "model/sam/image_encoder.h"
#include <yaml-cpp/yaml.h>

using namespace sam;

ImageEncoder::ImageEncoder(const std::string &yaml_file) : m_input_size_(1024, 1024), m_output_size_(64, 64) 
{
    YAML::Node yaml_node = YAML::LoadFile(yaml_file);

    std::string model_path = yaml_node["model_path"].as<std::string>();
    std::string framework_type = yaml_node["framework"].as<std::string>();

    if (!Init(model_path, framework_type)) exit(0);

    config_.input_len["image"] = 3 * m_input_size_.height * m_input_size_.width;
    config_.output_len["image_embeddings"] = 256 * m_output_size_.height * m_output_size_.width;
    config_.is_dynamic = false;
    Status status = framework_->Init(config_);
    if (status != Status::SUCCESS) {
        std::cout << "Failed to init framework" << std::endl;
        exit(0);
    }
}

ImageEncoder::~ImageEncoder() {
    std::cout << "Destruct image encoder" << std::endl;
}

void ImageEncoder::preprocess(const cv::Mat &input_image, cv::Mat &output_image) {
    cv::dnn::blobFromImage(input_image, output_image, 1 / 57.f, cv::Size(), cv::Scalar(123.675, 116.28, 103.53), true, false, CV_32F);
}

void ImageEncoder::forward(const cv::Mat &image, IOTensor& features) {
    std::unordered_map<std::string, IOTensor> input, output;

    cv::Mat nchw;
    preprocess(image, nchw);

    input["image"] = IOTensor();
    input["image"].resize(nchw.total() * nchw.elemSize());
    input["image"].shape = std::vector<int64_t>{1, 3, m_input_size_.height, m_input_size_.width};
    input["image"].data_type = DataType::FP32;
    memcpy(input["image"].data(), nchw.ptr<uint8_t>(), nchw.total() * nchw.elemSize());
    

    // 输出张量设置
    output["image_embeddings"] = IOTensor();
    output["image_embeddings"].data_type = DataType::FP32;
    output["image_embeddings"].shape = std::vector<int64_t>{1, 256, m_output_size_.height, m_output_size_.width};
    output["image_embeddings"].resize(config_.output_len["image_embeddings"] * sizeof(float));

    this->framework_->forward(input, output);

    features.resize(config_.output_len["image_embeddings"] * sizeof(float));
    memcpy(features.data(), output["image_embeddings"].data(), features.size());
    features.shape = std::vector<int64_t>{1, 256, 64, 64};
}