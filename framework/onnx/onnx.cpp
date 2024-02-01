#include "framework/onnx/onnx.h"

int TypeToSize(const ONNXTensorElementDataType& dataType) {
    switch (dataType) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return 4;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            return 2;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            return 4;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            return 1;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            return 1;
        default:
            std::cout << "Unknown data type " << dataType << std::endl;
            return 4;
    }
}

Status ONNXFramework::Init(Config config) {
    env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ONNX_DETECTION");
    session_options = Ort::SessionOptions();

    Ort::AllocatorWithDefaultOptions allocator;

#ifdef _WIN32
    std::wstring w_modelPath = utils::charToWstring(model_path.c_str());
    session = Ort::Session(env, w_modelPath.c_str(), sessionOptions);
#else
    session = new Ort::Session(env, config.model_path.c_str(), session_options);
#endif

    int input_num = session->GetInputCount();
    for (int i = 0; i < input_num; i++) {
        Ort::TypeInfo input_type_info = session->GetInputTypeInfo(i);
        std::vector<int64_t> input_tensor_shape = input_type_info.GetTensorTypeAndShapeInfo().GetShape();

        Binding binding;
        size_t size = 1;
        for (size_t j = 0; j < input_tensor_shape.size(); j++) {
            binding.dims.push_back(input_tensor_shape[j]);
            size *= input_tensor_shape[j];
        }

        binding.size = size;
        binding.dsize = TypeToSize(input_type_info.GetTensorTypeAndShapeInfo().GetElementType());

        Ort::AllocatedStringPtr input_name = session->GetInputNameAllocated(i, allocator);
        binding.name = input_name.get();
        input_bindings.push_back(binding);
        if (config.input_len[binding.name] != size) {
            std::cout << "Input size of " << binding.name << " mismatch the model file " << config.model_path << ". ("
                      << config.input_len[binding.name] << "!=" << size << ")" << std::endl;
            return Status::INIT_ERROR;
        }
    }

    std::cout << "Input: " << std::endl;
    for (const auto& binding : input_bindings) {
        std::cout << binding.name << ": " << binding.size << std::endl;
    }

    int output_num = session->GetOutputCount();
    for (int i = 0; i < output_num; i++) {
        Binding binding;

        Ort::TypeInfo output_type_info = session->GetOutputTypeInfo(i);
        std::vector<int64_t> output_tensor_shape = output_type_info.GetTensorTypeAndShapeInfo().GetShape();

        size_t size = 1;
        for (size_t j = 0; j < output_tensor_shape.size(); j++) {
            binding.dims.push_back(output_tensor_shape[j]);
            size *= output_tensor_shape[j];
        }
        binding.size = size;
        binding.dsize = TypeToSize(output_type_info.GetTensorTypeAndShapeInfo().GetElementType());

        Ort::AllocatedStringPtr output_name = session->GetOutputNameAllocated(i, allocator);
        binding.name = output_name.get();
        output_bindings.push_back(binding);

        if (config.output_len[binding.name] != size) {
            std::cout << "Output size of " << binding.name << " mismatch the model file " << config.model_path << ". ("
                      << config.output_len[binding.name] << "!=" << size << ")" << std::endl;
            return Status::INIT_ERROR;
        }

    }

    std::cout << "Output: " << std::endl;
    for (const auto& binding : output_bindings) {
        std::cout << binding.name << ": " << binding.size << std::endl;
    }

    return Status::SUCCESS;
}

ONNXFramework::~ONNXFramework() {
    delete session;
}

Status ONNXFramework::forward(const std::unordered_map<std::string, IOTensor>& input,
                              std::unordered_map<std::string, IOTensor>& output) {
    std::vector<Ort::Value> input_tensors;
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    std::vector<const char*> input_names;
    for (const auto& binding : input_bindings) {
        const std::string input_name = binding.name;
        input_names.emplace_back(input_name.c_str());
        if (input.find(input_name) == input.end()) {
            std::cout << "Cannot find " << input_name << " from the input tensors!" << std::endl;
            return Status::INFERENCE_ERROR;
        }

        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info, (float*)input.at(input_name).data(), binding.size, binding.dims.data(), binding.dims.size()));
    }

    std::vector<const char*> output_names;
    std::vector<Ort::Value> output_tensors;
    for (const auto& binding : output_bindings) {
        output_names.emplace_back(binding.name.c_str());
        if (output.find(binding.name) == output.end()) {
            std::cout << "Cannot find " << binding.name << " from the input tensors!" << std::endl;
            return Status::INFERENCE_ERROR;
        }
        output[binding.name].resize(sizeof(float) * binding.size);
        output_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info, (float*)output[binding.name].data(), binding.size, binding.dims.data(), binding.dims.size()));
    }

    this->session->Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_names.size(),
                          output_names.data(), output_tensors.data(), output_names.size());
    return Status::SUCCESS;
}