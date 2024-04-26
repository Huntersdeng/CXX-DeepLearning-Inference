#include "model/yolo/yolo.h"
#include <yaml-cpp/yaml.h>

YOLO::YOLO(const std::string &yaml_file) {
    YAML::Node yaml_node = YAML::LoadFile(yaml_file);

    std::string model_path = yaml_node["model_path"].as<std::string>();
    std::string framework_type = yaml_node["framework"].as<std::string>();
    if (!Init(model_path, framework_type)) exit(0);

    std::vector<long> input_size = yaml_node["input_size"].as<std::vector<long>>();
    m_input_size_.width = input_size.at(0);
    m_input_size_.height = input_size.at(1);
    topk_ = yaml_node["topk"].as<int>();
    
    with_nms_ = yaml_node["with_nms"].as<bool>();
    if (!with_nms_) {  
        m_conf_thres_ = yaml_node["conf_thres"].as<float>();
        m_nms_thres_ = yaml_node["nms_thres"].as<float>();  
        m_grid_num_ = 0;
        for (int i = 0; i < 3; i++)
        {
            m_grid_num_ += (m_input_size_.width / strides[i]) * (m_input_size_.height / strides[i]);
        }
        config_.input_len["images"] = 3 * m_input_size_.height * m_input_size_.width;
        config_.output_len["output"] = m_grid_num_ * 6;
        config_.is_dynamic = false;
    } else {
        config_.input_len["images"] = 3 * m_input_size_.height * m_input_size_.width;
        config_.output_len["num_dets"] = 1;
        config_.output_len["bboxes"] = 4 * topk_;
        config_.output_len["scores"] = topk_;
        config_.output_len["labels"] = topk_;
    }

    config_.is_dynamic = false;
    Status status = framework_->Init(config_);
    if (status != Status::SUCCESS) {
        std::cout << "Failed to init framework" << std::endl;
        exit(0);
    }
}

YOLO::~YOLO()
{
    std::cout << "Destruct yolov8" << std::endl;
}

void YOLO::preprocess(const cv::Mat &input_image, cv::Mat &output_image) {
    cv::Mat mask;
    this->pparam_ = Letterbox(input_image, mask, m_input_size_);
    cv::dnn::blobFromImage(mask, output_image, 1 / 255.f, cv::Size(), cv::Scalar(0, 0, 0), false, false, CV_32F);
}

void YOLO::detect(const cv::Mat &image, std::vector<Object> &objs) {
    std::unordered_map<std::string, IOTensor> input, output;

    // 输入tensor设置
    cv::Mat nchw;
    preprocess(image, nchw);

    if (!with_nms_) {
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
    } else {
        input["images"] = IOTensor();
        input["images"].shape = std::vector<int64_t>{1, 3, m_input_size_.height, m_input_size_.width};
        input["images"].data_type = DataType::FP32;
        input["images"].resize(nchw.total() * nchw.elemSize());
        memcpy(input["images"].data(), nchw.ptr<uint8_t>(), nchw.total() * nchw.elemSize());

        // 输出张量设置
        output["num_dets"] = IOTensor();
        output["num_dets"].shape = std::vector<int64_t>{1, 1};
        output["num_dets"].data_type = DataType::INT32;
        output["num_dets"].resize(config_.output_len["num_dets"] * sizeof(int));

        output["bboxes"] = IOTensor();
        output["bboxes"].shape = std::vector<int64_t>{1, 100, 4};
        output["bboxes"].data_type = DataType::FP32;
        output["bboxes"].resize(config_.output_len["bboxes"] * sizeof(float));

        output["scores"] = IOTensor();
        output["scores"].shape = std::vector<int64_t>{1, 100};
        output["scores"].data_type = DataType::FP32;
        output["scores"].resize(config_.output_len["scores"] * sizeof(float));

        output["labels"] = IOTensor();
        output["labels"].shape = std::vector<int64_t>{1, 100};
        output["labels"].data_type = DataType::INT32;
        output["labels"].resize(config_.output_len["labels"] * sizeof(int));
    }

    this->framework_->forward(input, output);
    postprocess(output, objs);
    
}

void YOLO::postprocess(const std::unordered_map<std::string, IOTensor> &output, std::vector<Object> &objs) {
    if (!with_nms_) {
        postprocess_with_nms(output, objs);
    } else {
        postprocess_without_nms(output, objs);
    }
}

void YOLO::postprocess_without_nms(const std::unordered_map<std::string, IOTensor> &output, std::vector<Object> &objs) {
    objs.clear();
    int *const num_dets = (int*)(output.at("num_dets").data());
    float *const boxes = (float *)(output.at("bboxes").data());
    float *scores = (float *)(output.at("scores").data());
    int *labels = (int*)(output.at("labels").data());
    auto &dw = this->pparam_.dw;
    auto &dh = this->pparam_.dh;
    auto &width = this->pparam_.width;
    auto &height = this->pparam_.height;
    auto &ratio = this->pparam_.ratio;
    for (int i = 0; i < num_dets[0]; i++)
    {
        float *ptr = boxes + i * 4;

        float x0 = *ptr++ - dw;
        float y0 = *ptr++ - dh;
        float x1 = *ptr++ - dw;
        float y1 = *ptr - dh;

        x0 = clamp(x0 * ratio, 0.f, width);
        y0 = clamp(y0 * ratio, 0.f, height);
        x1 = clamp(x1 * ratio, 0.f, width);
        y1 = clamp(y1 * ratio, 0.f, height);
        Object obj;
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = x1 - x0;
        obj.rect.height = y1 - y0;
        obj.prob = *(scores + i);
        obj.label = *(labels + i);
        objs.push_back(obj);
    }
}

void YOLO::postprocess_with_nms(const std::unordered_map<std::string, IOTensor> &output, std::vector<Object> &objs)
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
        if (cnt >= topk_)
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