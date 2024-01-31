#pragma once

#include <onnxruntime_cxx_api.h>

#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>

#include "common/framework.h"

int TypeToSize(const ONNXTensorElementDataType &dataType);

class ONNXFramework : public BaseFramework {
   public:
    ONNXFramework(std::string model_path);
    ~ONNXFramework();
    STATUS forward(const std::unordered_map<std::string, IOTensor> &input,
                 std::unordered_map<std::string, IOTensor> &output) override;

   private:
    Ort::Env env{nullptr};
    Ort::SessionOptions session_options{nullptr};
    Ort::Session session{nullptr};
    std::vector<float *> temp_output_ptrs;
};