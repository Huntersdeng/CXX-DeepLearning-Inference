#pragma once

#include <unordered_map>
#include <vector>

#include "common/common.h"

using IOTensor = std::vector<uint8_t>;

enum Status { SUCCESS = 0, INIT_ERROR = -1, INFERENCE_ERROR = -2};

struct Config {
    std::string model_path;
    std::map<std::string, int64_t> input_len;
    std::map<std::string, int64_t> output_len;
};

class BaseFramework {
   public:
    BaseFramework() {}
    virtual ~BaseFramework() {}
    virtual Status Init(Config config) = 0;
    virtual Status forward(const std::unordered_map<std::string, IOTensor> &input,
                           std::unordered_map<std::string, IOTensor> &output) = 0;

   protected:
    std::vector<Binding> input_bindings;
    std::vector<Binding> output_bindings;
};