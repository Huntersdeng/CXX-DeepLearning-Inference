#include "framework/tensorrt.h"

using namespace det;

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

int get_size_by_dims(const nvinfer1::Dims &dims) {
    int size = 1;
    for (int i = 0; i < dims.nbDims; i++) {
        size *= dims.d[i];
    }
    return size;
}

int type_to_size(const nvinfer1::DataType &dataType) {
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

TensorRTFramework::TensorRTFramework(const std::string &engine_file_path) {
    // 读取模型文件
    std::ifstream file(engine_file_path, std::ios::binary);
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

    this->num_bindings = this->engine->getNbBindings();
    for (int i = 0; i < this->num_bindings; ++i) {
        Binding binding;
        nvinfer1::Dims dims;
        nvinfer1::DataType dtype = this->engine->getBindingDataType(i);
        std::string name = this->engine->getBindingName(i);
        binding.name = name;
        binding.dsize = type_to_size(dtype);

        bool IsInput = engine->bindingIsInput(i);
        if (IsInput) {
            this->num_inputs += 1;
            dims = this->engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->input_bindings.push_back(binding);
            // set max opt shape
            this->context->setBindingDimensions(i, dims);
        } else {
            dims = this->context->getBindingDimensions(i);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->output_bindings.push_back(binding);
            this->num_outputs += 1;
        }
    }

    make_pipe(true);
}

TensorRTFramework::~TensorRTFramework() {
    std::cout << "Destruct tensorrt" << std::endl;
    this->context->destroy();
    this->engine->destroy();
    this->runtime->destroy();
    cudaStreamDestroy(this->stream);
    for (auto &ptr : this->device_ptrs) {
        CHECK(cudaFree(ptr));
    }

    for (auto &ptr : this->host_ptrs) {
        CHECK(cudaFreeHost(ptr));
    }
}

void TensorRTFramework::make_pipe(bool warmup) {
    for (auto &bindings : this->input_bindings) {
        void *d_ptr;
        CHECK(cudaMalloc(&d_ptr, bindings.size * bindings.dsize));
        this->device_ptrs.push_back(d_ptr);
    }

    for (auto &bindings : this->output_bindings) {
        void *d_ptr, *h_ptr;
        size_t size = bindings.size * bindings.dsize;
        CHECK(cudaMalloc(&d_ptr, size));
        CHECK(cudaHostAlloc(&h_ptr, size, 0));
        this->device_ptrs.push_back(d_ptr);
        this->host_ptrs.push_back(h_ptr);
    }

    if (warmup) {
        for (int i = 0; i < 10; i++) {
            for (auto &bindings : this->input_bindings) {
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

void TensorRTFramework::set_input(const cv::Mat &image) {
    auto &in_binding = this->input_bindings[0];
    auto width = in_binding.dims.d[3];
    auto height = in_binding.dims.d[2];

    this->context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, height, width}});

    CHECK(cudaMemcpyAsync(this->device_ptrs[0], image.ptr<float>(), image.total() * image.elemSize(),
                          cudaMemcpyHostToDevice, this->stream));
}

void TensorRTFramework::infer() {
    this->context->enqueueV2(this->device_ptrs.data(), this->stream, nullptr);
    for (int i = 0; i < this->num_outputs; i++) {
        size_t osize = this->output_bindings[i].size * this->output_bindings[i].dsize;
        CHECK(cudaMemcpyAsync(this->host_ptrs[i], this->device_ptrs[i + this->num_inputs], osize,
                              cudaMemcpyDeviceToHost, this->stream));
    }
    cudaStreamSynchronize(this->stream);
}

void TensorRTFramework::forward(const cv::Mat &image, std::vector<void *> &output) {
    this->set_input(image);
    this->infer();
    output = this->host_ptrs;
}