#pragma once
#include <fstream>

#include "model/base/ocr_model.h"

class AttnModel : public OcrModel {
   public:
    AttnModel() = delete;
    explicit AttnModel(const std::string &model_path, const std::string framework_type, cv::Size input_size, size_t input_channel,
                  size_t output_size, const std::string alphabet) : OcrModel(model_path, framework_type, input_size, input_channel, output_size, alphabet) {}
    explicit AttnModel(const std::string &yaml_file) : OcrModel(yaml_file) {}
    ~AttnModel() {}

    std::string detect(const cv::Mat &image) override;

   protected:
    std::string postprocess(const std::unordered_map<std::string, IOTensor> &output) override;
};