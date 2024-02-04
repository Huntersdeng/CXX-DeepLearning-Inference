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
    explicit OcrModel() {}; 
    virtual ~OcrModel() {};
    virtual std::string detect(const cv::Mat &image) = 0;
protected:
    virtual std::string postprocess(const std::unordered_map<std::string, IOTensor> &output) = 0;
    std::string alphabet_;
};