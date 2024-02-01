#pragma once

#include <onnxruntime_cxx_api.h>

#include <fstream>
#include <string>

#include "framework/framework.h"

int TypeToSize(const ONNXTensorElementDataType &dataType);

class ONNXFramework : public BaseFramework {
   public:
    ONNXFramework() {}
    ~ONNXFramework();
    Status Init(Config config) override;
    Status forward(const std::unordered_map<std::string, IOTensor> &input,
                 std::unordered_map<std::string, IOTensor> &output) override;

   private: 
    Ort::Env env{nullptr};
    Ort::SessionOptions session_options{nullptr};
    Ort::Session *session{nullptr};
    std::vector<float *> temp_output_ptrs;
};