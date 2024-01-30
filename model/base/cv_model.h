#pragma once
#include "framework/common/common.h"
#include "framework/common/framework.h"
#include "framework/onnx/onnx.h"
#include "framework/config.h"

#ifdef USE_TENSORRT
    #include "framework/tensorrt.h"
#endif

class CvModel
{
public:
    explicit CvModel(const std::string &model_path);
    virtual ~CvModel() {};
    virtual void detect(const cv::Mat &image, std::vector<Object> &objs) = 0;
protected:
    virtual void postprocess(const std::vector<void*> output, std::vector<Object> &objs) = 0;

    cv::Size m_input_size_ = {640, 640};
    PreParam pparam;
    std::shared_ptr<BaseFramework> framework;
};