#include "framework/tensorrt/tensorrt.h"

void Logger::log(nvinfer1::ILogger::Severity severity, const char *msg) noexcept {
    if (severity > reportableSeverity) {
        return;
    }
    switch (severity) {
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
            std::cerr << "INTERNAL_ERROR: ";
            break;
        case nvinfer1::ILogger::Severity::kERROR:
            std::cerr << "ERROR: ";
            break;
        case nvinfer1::ILogger::Severity::kWARNING:
            std::cerr << "WARNING: ";
            break;
        case nvinfer1::ILogger::Severity::kINFO:
            std::cerr << "INFO: ";
            break;
        default:
            std::cerr << "VERBOSE: ";
            break;
    }
    std::cerr << msg << std::endl;
}

int TypeToSize(const nvinfer1::DataType &dataType) {
    switch (dataType) {
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kINT8:
            return 1;
        case nvinfer1::DataType::kBOOL:
            return 1;
        default:
            return 4;
    }
}

Status TensorRTFramework::Init(Config config) {
    // 读取模型文件
    std::ifstream file(config.model_path, std::ios::binary);
    assert(file.good());
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    char *trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    // 加载插件
    initLibNvInferPlugins(&this->gLogger, "");

    // 创建IRuntime对象
    this->runtime = nvinfer1::createInferRuntime(this->gLogger);
    assert(this->runtime != nullptr);

    // 反序列化engine文件，创建ICudaEngine对象
    this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
    assert(this->engine != nullptr);
    delete[] trtModelStream;

    // 初始化IExecutionContext对象
    this->context = this->engine->createExecutionContext();
    assert(this->context != nullptr);

    // 创建cudaStream_t对象
    cudaStreamCreate(&this->stream);

    this->is_dynamic = config.is_dynamic;

    this->num_bindings = this->engine->getNbIOTensors();
    for (int i = 0; i < this->num_bindings; ++i)
    {
        Binding binding;
        nvinfer1::Dims dims;
        std::string name = this->engine->getIOTensorName(i);
        nvinfer1::DataType dtype = this->engine->getTensorDataType(name.c_str());
        binding.name = name;
        binding.dsize = TypeToSize(dtype);

        nvinfer1::TensorIOMode io_mode = engine->getTensorIOMode(name.c_str());
        if (io_mode == nvinfer1::TensorIOMode::kINPUT)
        {
            in_index_[name] = this->num_inputs;
            this->num_inputs += 1;
            dims = this->engine->getProfileShape(name.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
            binding.size = 1;
            for (int i = 0; i < dims.nbDims; i++)
            {
                binding.size *= dims.d[i];
                binding.dims.push_back(dims.d[i]);
            }
            if (!is_dynamic && config.input_len[binding.name] != binding.size) {
                std::cout << "Input size of " << binding.name << " mismatch the model file " << config.model_path << ". ("
                        << config.input_len[binding.name] << "!=" << binding.size << ")" << std::endl;
                return Status::INIT_ERROR;
            }
            this->input_bindings.push_back(binding);
            // set max opt shape
            this->context->setInputShape(name.c_str(), dims);
            std::cout << "Input bind name: " << name << std::endl;
        }
        else if (io_mode == nvinfer1::TensorIOMode::kOUTPUT)
        {
            out_index_[name] = this->num_outputs;
            dims = this->context->getTensorShape(name.c_str());
            binding.size = 1;
            for (int i = 0; i < dims.nbDims; i++)
            {
                binding.size *= dims.d[i];
                binding.dims.push_back(dims.d[i]);
            }
            if (!is_dynamic && config.output_len[binding.name] != binding.size) {
                std::cout << "Output size of " << binding.name << " mismatch the model file " << config.model_path << ". ("
                        << config.output_len[binding.name] << "!=" << binding.size << ")" << std::endl;
                return Status::INIT_ERROR;
            }
            this->output_bindings.push_back(binding);
            this->num_outputs += 1;
            std::cout << "Output bind name: " << name << std::endl;
        }
    }
    make_pipe(true);
    return Status::SUCCESS;
}

TensorRTFramework::~TensorRTFramework() {
    delete this->context;
    delete this->engine;
    delete this->runtime;
    cudaStreamDestroy(this->stream);
    for (auto &ptr : this->device_ptrs)
    {
        CHECK(cudaFree(ptr));
    }

    for (auto &ptr : this->host_ptrs)
    {
        CHECK(cudaFreeHost(ptr));
    }
}

void TensorRTFramework::make_pipe(bool warmup) {
    for (auto &bindings : this->input_bindings)
    {
        void *d_ptr;
        CHECK(cudaMalloc(&d_ptr, bindings.size * bindings.dsize));
        this->device_ptrs.push_back(d_ptr);
        this->context->setTensorAddress(bindings.name.c_str(), d_ptr);
    }

    for (auto &bindings : this->output_bindings)
    {
        void *d_ptr, *h_ptr;
        size_t size = bindings.size * bindings.dsize;
        CHECK(cudaMalloc(&d_ptr, size));
        CHECK(cudaHostAlloc(&h_ptr, size, 0));
        this->device_ptrs.push_back(d_ptr);
        this->host_ptrs.push_back(h_ptr);
        this->context->setTensorAddress(bindings.name.c_str(), d_ptr);
    }

    if (warmup)
    {
        for (int i = 0; i < 10; i++)
        {
            for (auto &bindings : this->input_bindings)
            {
                size_t size = bindings.size * bindings.dsize;
                void *h_ptr = malloc(size);
                memset(h_ptr, 0, size);
                CHECK(cudaMemcpyAsync(this->device_ptrs[0], h_ptr, size, cudaMemcpyHostToDevice, this->stream));
                free(h_ptr);
            }
            this->infer();
        }
        printf("model warmup 10 times\n");
    }
}

bool TensorRTFramework::set_input(const std::unordered_map<std::string, IOTensor> &input) {
    for (auto &kv : input) {
        size_t idx = in_index_[kv.first];
        auto& binding = this->input_bindings[idx];
        if (input.find(binding.name) == input.end()) {
            std::cout << "Cannot find " << binding.name << " from the input tensors!" << std::endl;
            return false;
        }
        if (is_dynamic) {
            std::vector<int64_t> shape = input.at(binding.name).shape;
            nvinfer1::Dims dim;
            dim.nbDims = shape.size();
            for (size_t i = 0; i < dim.nbDims; i++) {
                dim.d[i] = shape[i];
            }
            context->setInputShape(binding.name.c_str(), dim);
        }
        CHECK(cudaMemcpyAsync(
            this->device_ptrs[idx], kv.second.data(), kv.second.size(), cudaMemcpyHostToDevice, this->stream));
    }
    return true;
}

bool TensorRTFramework::infer() {
    this->context->enqueueV3(this->stream);
    for (int i = 0; i < this->num_outputs; i++)
    {
        size_t osize = this->output_bindings[i].size * this->output_bindings[i].dsize;
        CHECK(cudaMemcpyAsync(
            this->host_ptrs[i], this->device_ptrs[i + this->num_inputs], osize, cudaMemcpyDeviceToHost, this->stream));
    }
    cudaStreamSynchronize(this->stream);
    return true;
}

Status TensorRTFramework::forward(const std::unordered_map<std::string, IOTensor> &input,
                           std::unordered_map<std::string, IOTensor> &output) {
    if (!this->set_input(input)) {
        return Status::INFERENCE_ERROR;
    }
    if (!this->infer()) {
        return Status::INFERENCE_ERROR;
    }
    for (auto &kv : output) {
        auto cur_idx = out_index_[kv.first];
        const auto& binding = this->output_bindings[cur_idx];
        memcpy(kv.second.data(), this->host_ptrs[cur_idx], kv.second.size());
    }
    return Status::SUCCESS;
}