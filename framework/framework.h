#pragma once

#include <unordered_map>
#include <vector>

#include "common/common.h"

struct IOTensor {
    std::vector<uint8_t> raw_data;
    std::vector<int64_t> shape;
    void resize(size_t size) {
        raw_data.resize(size);
    }

    size_t size() const {
        return raw_data.size();
    }

    uint8_t* data() {
        return raw_data.data();
    }

    const uint8_t* data() const{
        return raw_data.data();
    }
};

enum Status { SUCCESS = 0, INIT_ERROR = -1, INFERENCE_ERROR = -2};

struct Config {
    std::string model_path;
    std::map<std::string, int64_t> input_len;
    std::map<std::string, int64_t> output_len;
    bool is_dynamic;
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
    bool is_dynamic;
};