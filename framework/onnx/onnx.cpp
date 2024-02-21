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
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return 8;
        default:
            std::cout << "Unknown data type " << dataType << std::endl;
            return 4;
    }
}

Status ONNXFramework::Init(Config config) {
    is_dynamic = config.is_dynamic;
    env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ONNX_DETECTION");
    session_options = Ort::SessionOptions();

    Ort::AllocatorWithDefaultOptions allocator;

#ifdef _WIN32
    std::wstring w_modelPath = utils::charToWstring(model_path.c_str());
    session = Ort::Session(env, w_modelPath.c_str(), sessionOptions);
#else
    session = new Ort::Session(env, config.model_path.c_str(), session_options);
#endif

    std::cout << "Input: " << std::endl;
    int input_num = session->GetInputCount();
    for (int i = 0; i < input_num; i++) {
        Ort::TypeInfo input_type_info = session->GetInputTypeInfo(i);
        std::vector<int64_t> input_tensor_shape = input_type_info.GetTensorTypeAndShapeInfo().GetShape();

        Binding binding;
        int64_t size = 1;
        for (size_t j = 0; j < input_tensor_shape.size(); j++) {
            binding.dims.push_back(input_tensor_shape[j]);
            size *= input_tensor_shape[j];
        }

        if (size <= 0) {
            size = config.input_len[binding.name];
        }

        binding.size = size;
        binding.dsize = TypeToSize(input_type_info.GetTensorTypeAndShapeInfo().GetElementType());

        Ort::AllocatedStringPtr input_name = session->GetInputNameAllocated(i, allocator);
        binding.name = input_name.get();
        input_bindings.push_back(binding);
        std::cout << binding.name << ": [";
        for (size_t j = 0; j < input_tensor_shape.size(); j++) {
            std::cout << input_tensor_shape[j] << ",";
        }
        std::cout << "]" << std::endl;

        if (!is_dynamic && config.input_len[binding.name] != size) {
            std::cout << "Input size of " << binding.name << " mismatch the model file " << config.model_path << ". ("
                      << config.input_len[binding.name] << "!=" << size << ")" << std::endl;
            return Status::INIT_ERROR;
        }
    }

    std::cout << "Output: " << std::endl;
    int output_num = session->GetOutputCount();
    for (int i = 0; i < output_num; i++) {
        Binding binding;

        Ort::TypeInfo output_type_info = session->GetOutputTypeInfo(i);
        std::vector<int64_t> output_tensor_shape = output_type_info.GetTensorTypeAndShapeInfo().GetShape();

        Ort::AllocatedStringPtr output_name = session->GetOutputNameAllocated(i, allocator);
        binding.name = output_name.get();

        int64_t size = 1;
        for (size_t j = 0; j < output_tensor_shape.size(); j++) {
            binding.dims.push_back(output_tensor_shape[j]);
            size *= output_tensor_shape[j];
        }

        if (size <= 0) {
            size = config.output_len[binding.name];
        }

        binding.size = size;
        binding.dsize = TypeToSize(output_type_info.GetTensorTypeAndShapeInfo().GetElementType());

        output_bindings.push_back(binding);

        std::cout << binding.name << ": [";
        for (size_t j = 0; j < output_tensor_shape.size(); j++) {
            std::cout << output_tensor_shape[j] << ",";
        }
        std::cout << "]" << std::endl;

        if (!is_dynamic && config.output_len[binding.name] != size) {
            std::cout << "Output size of " << binding.name << " mismatch the model file " << config.model_path << ". ("
                      << config.output_len[binding.name] << "!=" << size << ")" << std::endl;
            return Status::INIT_ERROR;
        }

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
        input_names.emplace_back(binding.name.c_str());
        if (input.find(input_name) == input.end()) {
            std::cout << "Cannot find " << input_name << " from the input tensors!" << std::endl;
            return Status::INFERENCE_ERROR;
        }

        size_t size = 1;
        if (!is_dynamic) {
            size = binding.size;
        } else {
            for (size_t i = 0; i < input.at(input_name).shape.size(); i++) {
                size *= input.at(input_name).shape[i];
            }
        }
        if (input.at(input_name).data_type == DataType::INT32) {
            input_tensors.push_back(Ort::Value::CreateTensor<int>(
                memory_info, (int*)input.at(input_name).data(), size, input.at(input_name).shape.data(), input.at(input_name).shape.size()));
        } else if (input.at(input_name).data_type == DataType::FP32) {
            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                memory_info, (float*)input.at(input_name).data(), size, input.at(input_name).shape.data(), input.at(input_name).shape.size()));
        } else {
            std::cout << "Error occur when Ort::Value::CreateTensor" << std::endl;
        }
        
    }

    std::vector<const char*> output_names;
    for (const auto& binding : output_bindings) {
        output_names.emplace_back(binding.name.c_str());
        if (output.find(binding.name) == output.end()) {
            std::cout << "Cannot find " << binding.name << " from the input tensors!" << std::endl;
            return Status::INFERENCE_ERROR;
        }
    }

    std::vector<Ort::Value> output_tensors = this->session->Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_names.size(),
                          output_names.data(), output_names.size());
    
    for (size_t i = 0; i < output_tensors.size(); ++i){
        size_t element_size = TypeToSize(output_tensors[i].GetTensorTypeAndShapeInfo().GetElementType());
        size_t count = output_tensors[i].GetTensorTypeAndShapeInfo().GetElementCount();
        output[output_names[i]].resize(element_size * count);
        memcpy(output[output_names[i]].data(), output_tensors[i].GetTensorData<uint8_t>(), element_size * count);
        output[output_names[i]].shape = output_tensors[i].GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "Shape of " << output_names[i] << ": [";
        for (int64_t j : output[output_names[i]].shape) {
            std::cout << j << ",";
        }
        std::cout << "]" << std::endl;
    }
    return Status::SUCCESS;
}