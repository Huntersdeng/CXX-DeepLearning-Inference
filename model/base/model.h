#pragma once
#include "common/common.h"
#include "framework/framework.h"
#include "framework/onnx/onnx.h"
#include "framework/config.h"

#ifdef USE_TENSORRT
    #include "framework/tensorrt/tensorrt.h"
#endif

#ifdef USE_RKNN
    #include "framework/rknn/rknn.h"
#endif

class Model
{
public:
    explicit Model() {}; 
    virtual ~Model() {};
protected:
    bool Init(const std::string &model_path, const std::string &framework_type);
    Config config_;
    std::shared_ptr<BaseFramework> framework_;
};