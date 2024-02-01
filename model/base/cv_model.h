#pragma once
#include "common/common.h"
#include "framework/framework.h"
#include "framework/onnx/onnx.h"
#include "framework/config.h"

#ifdef USE_TENSORRT
    #include "framework/tensorrt/tensorrt.h"
#endif

class CvModel
{
public:
    explicit CvModel() {}; 
    virtual ~CvModel() {};
    virtual void detect(const cv::Mat &image, std::vector<Object> &objs) = 0;
protected:
    virtual void postprocess(const std::unordered_map<std::string, IOTensor> &output, std::vector<Object> &objs) = 0;

    Config config_;
    std::shared_ptr<BaseFramework> framework_;
};