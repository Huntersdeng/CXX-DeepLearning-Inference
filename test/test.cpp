#include "onnxruntime_cxx_api.h"

int main() {
    // Allocate ONNXRuntime session
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Env env;
    Ort::Session session{env, ORT_TSTR("../weights/ocr/best-train-abinet.onnx"), Ort::SessionOptions{nullptr}};

    // Allocate model inputs: fill in shape and size
    std::array<float, 12288> input;
    std::array<int64_t, 4> input_shape{1, 3, 32, 128};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(), input_shape.data(), input_shape.size());
    const char* input_names[] = {"images"};

    // Allocate model outputs: fill in shape and size
    std::array<int64_t, 26> output;
    std::array<int64_t, 3> output_shape{1, 26, 1};
    Ort::Value output_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, output.data(), output.size(), output_shape.data(), output_shape.size());
    const char* output_names[] = {"output"};

    // Run the model
    session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, &output_tensor, 1);
    return 0;
}   
 