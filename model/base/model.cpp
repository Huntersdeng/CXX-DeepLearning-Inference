#include "model/base/model.h"

bool Model::Init(const std::string &model_path, const std::string &framework_type) {
    config_.model_path = model_path;
    if (framework_type == "TensorRT")
    {   
    #ifdef USE_TENSORRT
        framework_ = std::make_shared<TensorRTFramework>();
    #else
        std::cout << "Framework " << framework_type << " not implemented" <<std::endl;
        return false;
    #endif
    }
    else if (framework_type == "ONNX")
    {
        framework_ = std::make_shared<ONNXFramework>();
    }
    else if (framework_type == "RKNN")
    {
    #ifdef USE_RKNN
        framework_ = std::make_shared<RknnFramework>();
    #else
        std::cout << "Framework " << framework_type << " not implemented" <<std::endl;
        return false;
    #endif
    }
    else
    {
        std::cout << "Framework " << framework_type << " not implemented" <<std::endl;
        return false;
    }
    return true;
}