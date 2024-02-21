#include "model/clip/image_encoder.h"
#include <yaml-cpp/yaml.h>
#include <assert.h>

using namespace clip;

ImageEncoder::ImageEncoder(const std::string &model_path, const std::string framework_type)
    : m_input_size_(224, 224), m_output_size_(512) {
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

    config_.input_len["IMAGE"] = 3 * m_input_size_.height * m_input_size_.width;
    config_.output_len["IMAGE_EMBEDDING"] = m_output_size_;
    config_.is_dynamic = true;
    Status status = framework_->Init(config_);
    if (status != Status::SUCCESS) {
        std::cout << "Failed to init framework" << std::endl;
        exit(0);
    }
}

ImageEncoder::ImageEncoder(const std::string &yaml_file) : m_input_size_(224, 224), m_output_size_(512) 
{
    YAML::Node yaml_node = YAML::LoadFile(yaml_file);

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

    config_.input_len["IMAGE"] = 3 * m_input_size_.height * m_input_size_.width;
    config_.output_len["IMAGE_EMBEDDING"] = m_output_size_;
    config_.is_dynamic = true;
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
    int h = input_image.rows;
    int w = input_image.cols;
    int resized_h, resized_w;
    if (h < w) {
        resized_h = 224;
        resized_w = int(224 * w / h);
    } else {
        resized_w = 224;
        resized_h = int(resized_w * h / w);
    }
    cv::Mat resized_img;
    cv::resize(input_image, resized_img, cv::Size(resized_w, resized_h));

    int y_from = (resized_h - 224) / 2;
    int x_from = (resized_w - 224) / 2;
    cv::Rect roi(x_from, y_from, 224, 224);
    resized_img = resized_img(roi);

    cv::Scalar mean(0.48145466*255, 0.4578275*255, 0.40821073*255);
    float std = (0.26862954 + 0.26130258 + 0.27577711) / 3 * 255;
    cv::dnn::blobFromImage(resized_img, output_image, 1 / std, cv::Size(), cv::Scalar(), true, false, CV_32F);
}

void ImageEncoder::forward(const std::vector<cv::Mat> &images, IOTensor& features) {
    std::unordered_map<std::string, IOTensor> input, output;

    input["IMAGE"] = IOTensor();
    input["IMAGE"].resize(images.size() * config_.input_len["IMAGE"] * sizeof(float));
    input["IMAGE"].shape = std::vector<int64_t>{static_cast<int64_t>(images.size()), 3, m_input_size_.height, m_input_size_.width};
    auto ptr = input["IMAGE"].data();
    for (const auto& image: images) {
        cv::Mat nchw;
        preprocess(image, nchw);
        assert(nchw.total() * nchw.elemSize() == config_.input_len["IMAGE"] * sizeof(float));
        memcpy(ptr, nchw.ptr<uint8_t>(), nchw.total() * nchw.elemSize());
        ptr += nchw.total() * nchw.elemSize();
    }

    // 输出张量设置
    output["IMAGE_EMBEDDING"] = IOTensor();
    output["IMAGE_EMBEDDING"].resize(images.size() * config_.output_len["IMAGE_EMBEDDING"] * sizeof(float));
    output["IMAGE_EMBEDDING"].shape = std::vector<int64_t>{static_cast<int64_t>(images.size()), config_.output_len["IMAGE_EMBEDDING"]};

    this->framework_->forward(input, output);

    features.resize(output["IMAGE_EMBEDDING"].size());
    memcpy(features.data(), output["IMAGE_EMBEDDING"].data(), features.size());
    features.shape = output["IMAGE_EMBEDDING"].shape;
}