#pragma once

#include <rknn_api.h>

#include <fstream>
#include <string>

#include "framework/framework.h"

int TypeToSize(const rknn_tensor_type &dataType);

class RknnFramework : public BaseFramework {
   public:
    RknnFramework() {}
    ~RknnFramework();
    Status Init(Config config) override;
    Status forward(const std::unordered_map<std::string, IOTensor> &input,
                 std::unordered_map<std::string, IOTensor> &output) override;

   private: 
    rknn_context rknn_ctx;
    rknn_tensor_attr* input_attrs_;
    rknn_tensor_attr* output_attrs_;
    std::unordered_map<std::string, int> in_index_;
    std::unordered_map<std::string, int> out_index_;
    bool is_quant_;
};