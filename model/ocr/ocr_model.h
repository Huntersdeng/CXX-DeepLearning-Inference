#pragma once
#include "model/base/model.h"
#include "common/common.h"

class OcrModel : public Model
{
public:
    OcrModel() = delete;
    explicit OcrModel(const std::string &yaml_file);
    virtual ~OcrModel();
    virtual std::string detect(const cv::Mat &image) = 0;
protected:
    virtual std::string postprocess(const std::unordered_map<std::string, IOTensor> &output) = 0;
    cv::Size m_input_size_ = {32, 100};
    size_t m_input_channel_ = 1;
    size_t m_output_length_ = 26;
    std::string alphabet_;
};