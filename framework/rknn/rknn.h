#pragma once

#include <rknn_api.h>

#include <fstream>
#include <string>

#include "framework/framework.h"

int TypeToSize(const rknn_tensor_type &dataType);

float sigmoid(float x);

float unsigmoid(float y);

float deqntAffineToF32(int8_t qnt, int32_t zp, float scale);

int32_t __clip(float val, float min, float max);

int8_t qntF32ToAffine(float f32, int32_t zp, float scale);

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