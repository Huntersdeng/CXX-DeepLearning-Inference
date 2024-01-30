#include "model/base/cv_model.h"


CvModel::CvModel(const std::string &model_path)
{
    std::string suffixStr = model_path.substr(model_path.find_last_of('.') + 1);
    if (suffixStr == "onnx")
    {
        framework = std::make_shared<ONNXFramework>(model_path);
    }
    #ifdef USE_TENSORRT
    else if (suffixStr == "engine")
    {
        framework = std::make_shared<TensorRTFramework>(model_path);
    }
    #endif
    else 
    {
        #ifdef USE_TENSORRT
            std::cerr << "Only support *.onnx and *.engine files" << std::endl;
        #else 
            std::cerr << "Only support *.onnx files" << std::endl;
        #endif
    }
}