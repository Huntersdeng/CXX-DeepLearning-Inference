#pragma once
#include <fstream>

#include "model/base/ocr_model.h"

class Crnn : public OcrModel {
   public:
    Crnn() = delete;
    explicit Crnn(const std::string &model_path, const std::string framework_type, cv::Size input_size,
                  size_t output_size, const std::string alphabet);
    explicit Crnn(const std::string &yaml_file);
    ~Crnn();

    std::string detect(const cv::Mat &image) override;

   protected:
    std::string postprocess(const std::unordered_map<std::string, IOTensor> &output) override;

   private:
    cv::Size m_input_size_ = {32, 100};
    size_t m_output_length_ = 26;
};