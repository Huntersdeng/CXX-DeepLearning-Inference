#pragma once
#include <fstream>

#include "model/base/ocr_model.h"

class AttnModel : public OcrModel {
   public:
    AttnModel() = delete;
    explicit AttnModel(const std::string &yaml_file) : OcrModel(yaml_file) {}
    ~AttnModel() {}

    std::string detect(const cv::Mat &image) override;

   protected:
    std::string postprocess(const std::unordered_map<std::string, IOTensor> &output) override;
};