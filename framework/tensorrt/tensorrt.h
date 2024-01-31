#pragma once

#include "NvInferPlugin.h"
#include <fstream>
#include "framework/framework.h"
#include "common.h"

class Logger : public nvinfer1::ILogger
{
public:
    nvinfer1::ILogger::Severity reportableSeverity;

    explicit Logger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO) : reportableSeverity(severity)
    {
    }

    void log(nvinfer1::ILogger::Severity severity, const char *msg) noexcept override;
};

int type_to_size(const nvinfer1::DataType &dataType);

class TensorRTFramework: public BaseFramework
{
public:
    explicit TensorRTFramework(const std::string &engine_file_path);
    virtual ~TensorRTFramework();
    void forward(void* input, std::vector<void *> &output) override;

private:
    void make_pipe(bool warmup = true);
    void set_input(const cv::Mat &image);
    void infer();

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

    det::PreParam pparam;
};