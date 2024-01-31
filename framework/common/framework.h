#pragma once

#include <unordered_map>
#include <vector>

#include "common/common.h"

using IOTensor = std::vector<uint8_t>;

enum STATUS {
    SUCCESS = 0,
    INPUT_KEY_ERROR = -1,
    OUTPUT_KEY_ERROR = -2
};

class BaseFramework {
   public:
    BaseFramework() {}
    virtual ~BaseFramework() {}
    virtual STATUS forward(const std::unordered_map<std::string, IOTensor> &input,
                         std::unordered_map<std::string, IOTensor> &output) = 0;

   protected:
    std::vector<Binding> input_bindings;
    std::vector<Binding> output_bindings;
};