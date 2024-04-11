#include "framework/rknn/rknn.h"

static void dump_tensor_attr(rknn_tensor_attr *attr) {
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

static int read_data_from_file(const char *path, char **out_data)
{
    FILE *fp = fopen(path, "rb");
    if(fp == NULL) {
        printf("fopen %s fail!\n", path);
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    int file_size = ftell(fp);
    char *data = (char *)malloc(file_size+1);
    data[file_size] = 0;
    fseek(fp, 0, SEEK_SET);
    if(file_size != fread(data, 1, file_size, fp)) {
        printf("fread %s fail!\n", path);
        free(data);
        fclose(fp);
        return -1;
    }
    if(fp) {
        fclose(fp);
    }
    *out_data = data;
    return file_size;
}

int TypeToSize(const rknn_tensor_type& dataType) {
    switch (dataType) {
        case RKNN_TENSOR_FLOAT32:
            return 4;
        case RKNN_TENSOR_FLOAT16:
            return 2;
        case RKNN_TENSOR_INT32:
            return 4;
        case RKNN_TENSOR_INT8:
            return 1;
        case RKNN_TENSOR_BOOL:
            return 1;
        case RKNN_TENSOR_INT64:
            return 8;
        default:
            std::cout << "Unknown data type " << dataType << std::endl;
            return 4;
    }
}

Status RknnFramework::Init(Config config) {
    is_dynamic = config.is_dynamic;
    int ret;
    int model_len = 0;
    char *model;
    rknn_context ctx = 0;

    // Load RKNN Model
    model_len = read_data_from_file(config.model_path.c_str(), &model);
    if (model == NULL)
    {
        printf("load_model fail!\n");
        return Status::INIT_ERROR;
    }

    ret = rknn_init(&ctx, model, model_len, 0, NULL);
    free(model);
    if (ret < 0)
    {
        printf("rknn_init fail! ret=%d\n", ret);
        return Status::INIT_ERROR;
    }

    // Get Model Input Output Number
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return Status::INIT_ERROR;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // Get Model Input Info
    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return Status::INIT_ERROR;
        }
        dump_tensor_attr(&(input_attrs[i]));
        Binding binding;
        binding.name = input_attrs[i].name;
        binding.size = input_attrs[i].n_elems;
        binding.dsize = TypeToSize(input_attrs[i].type);
        binding.dims = std::vector<int64_t>{input_attrs[i].dims[0], input_attrs[i].dims[1], input_attrs[i].dims[2], input_attrs[i].dims[3]};
        input_bindings.push_back(binding);
        in_index_[binding.name] = i;
        if (!is_dynamic && config.input_len[binding.name] != binding.size) {
            std::cout << "Input size of " << binding.name << " mismatch the model file " << config.model_path << ". ("
                      << config.input_len[binding.name] << "!=" << binding.size << ")" << std::endl;
            return Status::INIT_ERROR;
        }
    }

    // Get Model Output Info
    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return Status::INIT_ERROR;
        }
        dump_tensor_attr(&(output_attrs[i]));
        Binding binding;
        binding.name = output_attrs[i].name;
        binding.size = output_attrs[i].n_elems;
        binding.dsize = TypeToSize(output_attrs[i].type);
        binding.dims = std::vector<int64_t>{output_attrs[i].dims[0], output_attrs[i].dims[1], output_attrs[i].dims[2], output_attrs[i].dims[3]};
        output_bindings.push_back(binding);
        out_index_[binding.name] = i;
        if (!is_dynamic && config.output_len[binding.name] != binding.size) {
            std::cout << "Output size of " << binding.name << " mismatch the model file " << config.model_path << ". ("
                      << config.output_len[binding.name] << "!=" << binding.size << ")" << std::endl;
            return Status::INIT_ERROR;
        }
    }

    // Set to context
    rknn_ctx = ctx;

    // if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && output_attrs[0].type != RKNN_TENSOR_FLOAT16)
    // {
    //     is_quant_ = true;
    // }
    // else
    // {
    //     is_quant_ = false;
    // }

    input_attrs_ = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(input_attrs_, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    output_attrs_ = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(output_attrs_, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    uint32_t model_channel, model_height, model_width;
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("model is NCHW input fmt\n");
        model_channel = input_attrs[0].dims[1];
        model_height = input_attrs[0].dims[2];
        model_width = input_attrs[0].dims[3];
    }
    else
    {
        printf("model is NHWC input fmt\n");
        model_height = input_attrs[0].dims[1];
        model_width = input_attrs[0].dims[2];
        model_channel = input_attrs[0].dims[3];
    }
    printf("model input height=%d, width=%d, channel=%d\n",
           model_height, model_width, model_channel);

    return Status::SUCCESS;
}

RknnFramework::~RknnFramework() {
    if (rknn_ctx != 0)
    {
        rknn_destroy(rknn_ctx);
        rknn_ctx = 0;
    }
    if (input_attrs_ != NULL)
    {
        free(input_attrs_);
        input_attrs_ = NULL;
    }
    if (output_attrs_ != NULL)
    {
        free(output_attrs_);
        output_attrs_ = NULL;
    }
}

Status RknnFramework::forward(const std::unordered_map<std::string, IOTensor> &input, std::unordered_map<std::string, IOTensor> &output) {
    rknn_input rknn_input_tensors[input.size()];
    rknn_output rknn_output_tensors[output.size()];
    memset(rknn_input_tensors, 0, sizeof(rknn_input_tensors));
    memset(rknn_output_tensors, 0, sizeof(rknn_output_tensors));

    int ret = 0;

    for (auto &kv : input) {
        size_t idx = in_index_[kv.first];
        auto& binding = this->input_bindings[idx];
        if (input.find(binding.name) == input.end()) {
            std::cout << "Cannot find " << binding.name << " from the input tensors!" << std::endl;
            return Status::INFERENCE_ERROR;
        }
        rknn_input_tensors[0].index = idx;
        rknn_input_tensors[0].type = RKNN_TENSOR_UINT8;
        rknn_input_tensors[0].fmt = RKNN_TENSOR_NHWC;
        rknn_input_tensors[0].size = binding.size * binding.dsize;
        rknn_input_tensors[0].buf = (void*)kv.second.data();
    }

    ret = rknn_inputs_set(rknn_ctx, input_bindings.size(), rknn_input_tensors);
    if (ret < 0)
    {
        printf("rknn_input_set fail! ret=%d\n", ret);
        return Status::INFERENCE_ERROR;
    }

    ret = rknn_run(rknn_ctx, nullptr);
    if (ret < 0)
    {
        printf("rknn_run fail! ret=%d\n", ret);
        return Status::INFERENCE_ERROR;
    }

    memset(rknn_output_tensors, 0, sizeof(rknn_output_tensors));
    for (int i = 0; i < output_bindings.size(); i++)
    {
        rknn_output_tensors[i].index = i;
        rknn_output_tensors[i].want_float = false;
    }
    ret = rknn_outputs_get(rknn_ctx, output_bindings.size(), rknn_output_tensors, NULL);
    if (ret < 0)
    {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        return  Status::INFERENCE_ERROR;
    }

    for (auto &kv : output) {
        auto idx = out_index_[kv.first];
        const auto& binding = this->output_bindings[idx];
        kv.second.resize(binding.size);
        if (rknn_output_tensors[idx].size != binding.size) {
            return Status::INFERENCE_ERROR;
        }
        memcpy(kv.second.data(), rknn_output_tensors[idx].buf, kv.second.size());
        kv.second.zp = output_attrs_[idx].zp;
        kv.second.scale = output_attrs_[idx].scale;
    }

    return Status::SUCCESS;
}
