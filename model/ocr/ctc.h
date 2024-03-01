#pragma once
#include <fstream>

#include "model/base/ocr_model.h"

class CtcModel : public OcrModel {
   public:
    CtcModel() = delete;
    explicit CtcModel(const std::string &yaml_file): OcrModel(yaml_file) {}
    ~CtcModel() {}

    std::string detect(const cv::Mat &image) override;

   protected:
    std::string postprocess(const std::unordered_map<std::string, IOTensor> &output) override;

};