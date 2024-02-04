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

    Config config_;
    std::shared_ptr<BaseFramework> framework_;
};