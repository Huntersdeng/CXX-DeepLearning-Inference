#include "onnx/onnx.h"

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
        int size = 1;
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

        int size = 1;
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

        float* temp_output_ptr = (float*)malloc(sizeof(float) * size);
        assert(temp_output_ptr != nullptr);
        temp_output_ptrs.push_back(temp_output_ptr);
    }

    std::cout << "Output: " << std::endl;
    for (const auto& binding : output_bindings) {
        std::cout << binding.name << ": " << binding.size << std::endl;
    }

    return Status::SUCCESS;
}

ONNXFramework::~ONNXFramework() {
    for (size_t i = 0; i < temp_output_ptrs.size(); i++) {
        delete[] temp_output_ptrs[i];
    }
    delete session;
}

Status ONNXFramework::forward(const std::unordered_map<std::string, IOTensor>& input,
                              std::unordered_map<std::string, IOTensor>& output) {
    std::vector<Ort::Value> input_tensors;
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    std::vector<const char*> input_names(input_bindings.size());
    for (size_t i = 0; i < input_bindings.size(); i++) {
        const auto& binding = input_bindings[i];
        const std::string input_name = binding.name;
        input_names[i] = input_name.c_str();
        if (input.find(input_name) == input.end()) {
            std::cout << "Cannot find " << input_name << " from the input tensors!" << std::endl;
            return Status::INFERENCE_ERROR;
        }
        float const* blob = (float const*)input.at(input_name).data();
        size_t input_tensor_size = binding.size;
        std::vector<float> inputTensorValues(blob, blob + input_tensor_size);

        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info, inputTensorValues.data(), input_tensor_size, binding.dims.data(), binding.dims.size()));
    }

    std::vector<const char*> output_names(output_bindings.size());
    for (size_t i = 0; i < output_bindings.size(); i++) {
        output_names[i] = output_bindings[i].name.c_str();
    }

    std::vector<Ort::Value> output_tensors =
        this->session->Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_names.size(),
                          output_names.data(), output_names.size());

    for (size_t i = 0; i < output_tensors.size(); ++i) {
        if (output.find(output_names[i]) == output.end()) {
            std::cout << "Cannot find " << output_names[i] << " from the input tensors!" << std::endl;
            return Status::INFERENCE_ERROR;
        }
        auto* raw_output = output_tensors[i].GetTensorData<float>();
        size_t count = output_tensors[i].GetTensorTypeAndShapeInfo().GetElementCount();
        output[output_names[i]].resize(sizeof(float) * count);
        memcpy(output[output_names[i]].data(), raw_output, sizeof(float) * count);
    }
    return Status::SUCCESS;
}