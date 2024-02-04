#pragma once
#include "model/base/cv_model.h"
#include "common/common.h"
#include "framework/config.h"

#ifdef USE_TENSORRT
    #include "framework/tensorrt/tensorrt.h"
#endif

void ReadOCRLabel(const std::string& filename);

class OcrModel : public CvModel
{
public:
    OcrModel() = delete;
    explicit OcrModel(const std::string &model_path, const std::string framework_type, cv::Size input_size, size_t input_channel,
                  size_t output_size, const std::string alphabet);
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