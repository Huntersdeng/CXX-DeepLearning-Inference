#pragma once

#include "NvInferPlugin.h"
#include <fstream>
#include "framework/framework.h"
#include "common/common.h"

#define CHECK(call)                                                         \
    do                                                                      \
    {                                                                       \
        const cudaError_t error_code = call;                                \
        if (error_code != cudaSuccess)                                      \
        {                                                                   \
            printf("CUDA Error:\n");                                        \
            printf("    File:       %s\n", __FILE__);                       \
            printf("    Line:       %d\n", __LINE__);                       \
            printf("    Error code: %d\n", error_code);                     \
            printf("    Error text: %s\n", cudaGetErrorString(error_code)); \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

class Logger : public nvinfer1::ILogger
{
public:
    nvinfer1::ILogger::Severity reportableSeverity;

    explicit Logger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO) : reportableSeverity(severity)
    {
    }

    void log(nvinfer1::ILogger::Severity severity, const char *msg) noexcept override;
};

int TypeToSize(const nvinfer1::DataType &dataType);

class TensorRTFramework: public BaseFramework
{
public:
    explicit TensorRTFramework() {}
    virtual ~TensorRTFramework();
    Status Init(Config config) override;
    Status forward(const std::unordered_map<std::string, IOTensor> &input,
                           std::unordered_map<std::string, IOTensor> &output) override;

private:
    void make_pipe(bool warmup = true);
    bool set_input(const std::unordered_map<std::string, IOTensor> &input);
    bool infer();

    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IRuntime *runtime = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
    cudaStream_t stream = nullptr;
    Logger gLogger{nvinfer1::ILogger::Severity::kERROR};
    int num_bindings;
    int num_inputs = 0;
    int num_outputs = 0;
    std::vector<void *> host_ptrs;
    std::vector<void *> device_ptrs;
    std::unordered_map<std::string, int> in_index_;
    std::unordered_map<std::string, int> out_index_;

    PreParam pparam;
};