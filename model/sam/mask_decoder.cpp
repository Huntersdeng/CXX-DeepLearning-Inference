#include "model/sam/mask_decoder.h"
#include <yaml-cpp/yaml.h>

using namespace sam;

MaskDecoder::MaskDecoder(const std::string &model_path, const std::string framework_type)
    : features_shape{1, 256, 64, 64} {
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

    config_.input_len["image_embeddings"] =
        3 * features_shape[0] * features_shape[1] * features_shape[2] * features_shape[3];
    config_.input_len["point_coords"] = 10 * 2;
    config_.input_len["point_labels"] = 10;
    config_.input_len["mask_input"] = 1 * 1 * 256 * 256;
    config_.input_len["has_mask_input"] = 1;

    config_.output_len["iou_predictions"] = -1;
    config_.output_len["low_res_masks"] = -1;
    config_.is_dynamic = true;
    Status status = framework_->Init(config_);
    if (status != Status::SUCCESS) {
        std::cout << "Failed to init framework" << std::endl;
        exit(0);
    }
}

MaskDecoder::MaskDecoder(const std::string &yaml_file) : features_shape{1, 256, 64, 64}{
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

    config_.input_len["image_embeddings"] =
        features_shape[0] * features_shape[1] * features_shape[2] * features_shape[3];
    config_.input_len["point_coords"] = 10 * 2;
    config_.input_len["point_labels"] = 10;
    config_.input_len["mask_input"] = 1 * 1 * 256 * 256;
    config_.input_len["has_mask_input"] = 1;

    config_.output_len["iou_predictions"] = -1;
    config_.output_len["low_res_masks"] = -1;
    config_.is_dynamic = true;
    Status status = framework_->Init(config_);
    if (status != Status::SUCCESS) {
        std::cout << "Failed to init framework" << std::endl;
        exit(0);
    }
}

MaskDecoder::~MaskDecoder() { std::cout << "Destruct sam mask decoder" << std::endl; }


// The point labels may be
// | Point Label | Description |
// |:--------------------:|-------------|
// | 0 | Background point |
// | 1 | Foreground point |
// | 2 | Bounding box top-left |
// | 3 | Bounding box bottom-right |
void MaskDecoder::forward(const IOTensor &features, const std::vector<cv::Point2f> &image_point_coords,
                          const std::vector<float> &image_point_labels, cv::Mat& low_res_mask) {
    std::unordered_map<std::string, IOTensor> input, output;

    input["image_embeddings"] = IOTensor();
    input["image_embeddings"].shape = features_shape;
    input["image_embeddings"].resize(config_.input_len["image_embeddings"] * sizeof(float));
    memcpy(input["image_embeddings"].data(), features.data(), input["image_embeddings"].size());

    input["point_coords"] = IOTensor();
    input["point_coords"].shape = std::vector<int64_t>{1, static_cast<int64_t>(image_point_coords.size()), 2};
    input["point_coords"].resize(image_point_coords.size() * 2 * sizeof(float));
    std::vector<float> points;
    for (const auto& point: image_point_coords) {
        points.push_back(point.x);
        points.push_back(point.y);
    }
    memcpy(input["point_coords"].data(), points.data(), input["point_coords"].size());

    input["point_labels"] = IOTensor();
    input["point_labels"].shape = std::vector<int64_t>{1, static_cast<int64_t>(image_point_coords.size())};
    input["point_labels"].resize(image_point_coords.size() * sizeof(float));
    memcpy(input["point_labels"].data(), image_point_labels.data(), input["point_labels"].size());

    input["mask_input"] = IOTensor();
    input["mask_input"].shape = std::vector<int64_t>{1, 1, 256, 256};
    input["mask_input"].resize(256 * 256 * sizeof(float));

    input["has_mask_input"] = IOTensor();
    input["has_mask_input"].shape = std::vector<int64_t>{1};
    input["has_mask_input"].resize(sizeof(float));
    float has_mask_input = 0.0f;
    memcpy(input["has_mask_input"].data(), &has_mask_input, sizeof(float));

    // 输出张量设置
    output["iou_predictions"] = IOTensor();
    output["iou_predictions"].shape = std::vector<int64_t>{1, 4};
    output["iou_predictions"].resize(sizeof(float) * 4);

    output["low_res_masks"] = IOTensor();
    output["low_res_masks"].shape = std::vector<int64_t>{1, 4, 256, 256};
    output["low_res_masks"].resize(4 * 256 * 256 * sizeof(float));

    this->framework_->forward(input, output);

    low_res_mask = cv::Mat(256, 256, CV_32F, (float *)output.at("low_res_masks").data());
}