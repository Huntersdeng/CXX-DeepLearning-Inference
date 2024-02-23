#include "model/yolov8/yolov8.h"
#include <yaml-cpp/yaml.h>

YOLOv8::YOLOv8(const std::string &model_path,
                     const std::string framework_type,
                     cv::Size input_size,
                     float conf_thres,
                     float nms_thres) : m_input_size_(input_size),
                                         m_conf_thres_(conf_thres), m_nms_thres_(nms_thres) {
    config_.model_path = model_path;
    if (framework_type == "TensorRT")
    {   
    #ifdef USE_TENSORRT
        framework_ = std::make_shared<TensorRTFramework>();
    #else
        std::cout << "Framework " << framework_type << " not implemented" <<std::endl;
        exit(0);
    #endif
    }
    else if (framework_type == "ONNX")
    {
        framework_ = std::make_shared<ONNXFramework>();
    }
    else
    {
        std::cout << "Framework " << framework_type << " not implemented" <<std::endl;
        exit(0);
    }

    m_grid_num_ = 0;
    for (int i = 0; i < 3; i++)
    {
        m_grid_num_ += (m_input_size_.width / strides[i]) * (m_input_size_.height / strides[i]);
    }
    config_.input_len["images"] = 3 * m_input_size_.height * m_input_size_.width;
    config_.output_len["output"] = m_grid_num_ * 6;
    config_.is_dynamic = false;
    Status status = framework_->Init(config_);
    if (status != Status::SUCCESS) {
        std::cout << "Failed to init framework" << std::endl;
        exit(0);
    }
}

YOLOv8::YOLOv8(const std::string &yaml_file) {
    YAML::Node yaml_node = YAML::LoadFile(yaml_file);

    std::string model_path = yaml_node["model_path"].as<std::string>();
    std::string framework_type = yaml_node["framework"].as<std::string>();
    
    m_conf_thres_ = yaml_node["conf_thres"].as<float>();
    m_nms_thres_ = yaml_node["nms_thres"].as<float>();

    std::vector<long> input_size = yaml_node["input_size"].as<std::vector<long>>();
    m_input_size_.width = input_size.at(0);
    m_input_size_.height = input_size.at(1);

    config_.model_path = model_path;
    if (framework_type == "TensorRT")
    {   
    #ifdef USE_TENSORRT
        framework_ = std::make_shared<TensorRTFramework>();
    #else
        std::cout << "Framework " << framework_type << " not implemented" <<std::endl;
        exit(0);
    #endif
    }
    else if (framework_type == "ONNX")
    {
        framework_ = std::make_shared<ONNXFramework>();
    }
    else
    {
        std::cout << "Framework " << framework_type << " not implemented" <<std::endl;
        exit(0);
    }

    m_grid_num_ = 0;
    for (int i = 0; i < 3; i++)
    {
        m_grid_num_ += (m_input_size_.width / strides[i]) * (m_input_size_.height / strides[i]);
    }
    config_.input_len["images"] = 3 * m_input_size_.height * m_input_size_.width;
    config_.output_len["output"] = m_grid_num_ * 6;
    config_.is_dynamic = false;
    Status status = framework_->Init(config_);
    if (status != Status::SUCCESS) {
        std::cout << "Failed to init framework" << std::endl;
        exit(0);
    }
}

YOLOv8::~YOLOv8()
{
    std::cout << "Destruct yolov8" << std::endl;
}

void YOLOv8::preprocess(const cv::Mat &input_image, cv::Mat &output_image) {
    cv::Mat mask;
    this->pparam_ = Letterbox(input_image, mask, m_input_size_);
    cv::dnn::blobFromImage(mask, output_image, 1 / 255.f, cv::Size(), cv::Scalar(0, 0, 0), true, false, CV_32F);
}

void YOLOv8::detect(const cv::Mat &image, std::vector<Object> &objs) {
    std::unordered_map<std::string, IOTensor> input, output;

    // 输入tensor设置
    cv::Mat nchw;
    preprocess(image, nchw);

    input["images"] = IOTensor();
    input["images"].shape = std::vector<int64_t>{1, 3, m_input_size_.height, m_input_size_.width};
    input["images"].data_type = DataType::FP32;
    input["images"].resize(nchw.total() * nchw.elemSize());
    memcpy(input["images"].data(), nchw.ptr<uint8_t>(), nchw.total() * nchw.elemSize());
    

    // 输出张量设置
    output["output"] = IOTensor();
    output["output"].shape = std::vector<int64_t>{1, m_grid_num_, 6};
    output["output"].data_type = DataType::FP32;
    output["output"].resize(config_.output_len["output"] * sizeof(float));

    this->framework_->forward(input, output);
    postprocess(output, objs);
}

void YOLOv8::postprocess(const std::unordered_map<std::string, IOTensor> &output, std::vector<Object> &objs)
{
    objs.clear();
    auto num_anchors = m_grid_num_;

    auto &dw = this->pparam_.dw;
    auto &dh = this->pparam_.dh;
    auto &width = this->pparam_.width;
    auto &height = this->pparam_.height;
    auto &ratio = this->pparam_.ratio;

    std::vector<int> labels;
    std::vector<float> scores;
    std::vector<cv::Rect> bboxes;
    std::vector<int> indices;

    float * const outputs = (float *)output.at("output").data();

    for (int i = 0; i < num_anchors; i++)
    {
        float *ptr = outputs + i * 6;
        float score = *(ptr + 4);
        if (score > m_conf_thres_)
        {
            float x0 = *ptr++ - dw;
            float y0 = *ptr++ - dh;
            float x1 = *ptr++ - dw;
            float y1 = *ptr++ - dh;

            x0 = clamp(x0 * ratio, 0.f, width);
            y0 = clamp(y0 * ratio, 0.f, height);
            x1 = clamp(x1 * ratio, 0.f, width);
            y1 = clamp(y1 * ratio, 0.f, height);

            int label = *(++ptr);
            labels.push_back(label);
            scores.push_back(score);
            bboxes.push_back(cv::Rect_<float>(x0, y0, x1 - x0, y1 - y0));
        }
    }
    cv::dnn::NMSBoxes(bboxes, scores, m_conf_thres_, m_nms_thres_, indices);

    int cnt = 0;
    for (auto &i : indices)
    {
        if (cnt >= topk)
        {
            break;
        }
        cv::Rect tmp = bboxes[i];
        Object obj;
        obj.label = labels[i];
        obj.rect = tmp;
        obj.prob = scores[i];
        objs.push_back(obj);
        cnt += 1;
    }
}