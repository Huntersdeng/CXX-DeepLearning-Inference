#include "framework/framework.h"

#include "model/yolov8/yolov8_seg.h"
#include <yaml-cpp/yaml.h>

YOLOv8Seg::YOLOv8Seg(const std::string &model_path,
                     const std::string framework_type,
                     cv::Size input_size,
                     float conf_thres,
                     float nms_thres,
                     cv::Size seg_size,
                     int seg_channels) : m_input_size_(input_size),
                                         m_seg_size_(seg_size), m_seg_channels_(seg_channels),
                                         m_conf_thres_(conf_thres), m_nms_thres_(nms_thres)
{
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
    config_.output_len["outputs"] = m_grid_num_ * (m_seg_channels_ + 6);
    config_.output_len["proto"] = m_seg_channels_ * m_seg_size_.height * m_seg_size_.width;
    config_.is_dynamic = false;
    Status status = framework_->Init(config_);
    if (status != Status::SUCCESS) {
        std::cout << "Failed to init framework" << std::endl;
        exit(0);
    }
}

YOLOv8Seg::YOLOv8Seg(const std::string &yaml_file)
{
    YAML::Node yaml_node = YAML::LoadFile(yaml_file);

    std::string model_path = yaml_node["model_path"].as<std::string>();
    std::string framework_type = yaml_node["framework"].as<std::string>();
    
    m_conf_thres_ = yaml_node["conf_thres"].as<float>();
    m_nms_thres_ = yaml_node["nms_thres"].as<float>();

    std::vector<long> input_size = yaml_node["input_size"].as<std::vector<long>>();
    m_input_size_.width = input_size.at(0);
    m_input_size_.height = input_size.at(1);

    std::vector<long> seg_size = yaml_node["seg_size"].as<std::vector<long>>();
    m_seg_size_.width = seg_size.at(0);
    m_seg_size_.height = seg_size.at(1);

    m_seg_channels_ = yaml_node["seg_channels"].as<int>();

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
    config_.output_len["outputs"] = m_grid_num_ * (m_seg_channels_ + 6);
    config_.output_len["proto"] = m_seg_channels_ * m_seg_size_.height * m_seg_size_.width;
    config_.is_dynamic = false;
    Status status = framework_->Init(config_);
    if (status != Status::SUCCESS) {
        std::cout << "Failed to init framework" << std::endl;
        exit(0);
    }
}

YOLOv8Seg::~YOLOv8Seg()
{
    std::cout << "Destruct yolov8" << std::endl;
}

void YOLOv8Seg::preprocess(const cv::Mat &input_image, cv::Mat &output_image) {
    cv::Mat mask;
    this->pparam_ = Letterbox(input_image, mask, m_input_size_);
    cv::dnn::blobFromImage(mask, output_image, 1 / 255.f, cv::Size(), cv::Scalar(0, 0, 0), true, false, CV_32F);
}

void YOLOv8Seg::detect(const cv::Mat &image, std::vector<Object> &objs)
{
    std::unordered_map<std::string, IOTensor> input, output;

    // 输入tensor设置
    // auto start = std::chrono::system_clock::now();
    cv::Mat nchw;
    preprocess(image, nchw);

    input["images"] = IOTensor();
    input["images"].resize(nchw.total() * nchw.elemSize());
    memcpy(input["images"].data(), nchw.ptr<uint8_t>(), nchw.total() * nchw.elemSize());
    

    // 输出张量设置
    output["outputs"] = IOTensor();
    output["proto"] = IOTensor();
    output["outputs"].resize(config_.output_len["outputs"] * sizeof(float));
    output["proto"].resize(config_.output_len["proto"] * sizeof(float));
    // auto end = std::chrono::system_clock::now();
    // auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
    // std::cout << "Preprocess costs " << tc << " ms" << std::endl;

    // start = std::chrono::system_clock::now();
    this->framework_->forward(input, output);
    // end = std::chrono::system_clock::now();
    // tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
    // std::cout << "Inference costs " << tc << " ms" << std::endl;

    // start = std::chrono::system_clock::now();
    postprocess(output, objs);
    // end = std::chrono::system_clock::now();
    // tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
    // std::cout << "Postprocess costs " << tc << " ms" << std::endl;
}

void YOLOv8Seg::postprocess(const std::unordered_map<std::string, IOTensor> &output, std::vector<Object> &objs)
{
    objs.clear();
    auto seg_h = m_seg_size_.height;
    auto seg_w = m_seg_size_.width;
    auto input_h = m_input_size_.height;
    auto input_w = m_input_size_.width;
    auto num_anchors = m_grid_num_;
    auto num_channels = m_num_channels_;

    auto &dw = this->pparam_.dw;
    auto &dh = this->pparam_.dh;
    auto &width = this->pparam_.width;
    auto &height = this->pparam_.height;
    auto &ratio = this->pparam_.ratio;

    float * const outputs = (float *)output.at("outputs").data();
    cv::Mat protos = cv::Mat(m_seg_channels_, seg_h * seg_w, CV_32F, (float *)output.at("proto").data());
    assert(!protos.empty());

    std::vector<int> labels;
    std::vector<float> scores;
    std::vector<cv::Rect> bboxes;
    std::vector<cv::Mat> mask_confs;
    std::vector<int> indices;

    for (int i = 0; i < num_anchors; i++)
    {
        float *ptr = outputs + i * num_channels;
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
            cv::Mat mask_conf = cv::Mat(1, m_seg_channels_, CV_32F, ++ptr);
            mask_confs.push_back(mask_conf);
            labels.push_back(label);
            scores.push_back(score);
            bboxes.push_back(cv::Rect_<float>(x0, y0, x1 - x0, y1 - y0));
        }
    }
    cv::dnn::NMSBoxes(bboxes, scores, m_conf_thres_, m_nms_thres_, indices);

    cv::Mat masks;
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
        masks.push_back(mask_confs[i]);
        objs.push_back(obj);
        cnt += 1;
    }
    if (masks.empty())
    {
        // masks is empty
    }
    else
    {
        cv::Mat matmulRes = (masks * protos).t();
        cv::Mat maskMat = matmulRes.reshape(indices.size(), {seg_w, seg_h});

        std::vector<cv::Mat> maskChannels;
        cv::split(maskMat, maskChannels);
        int scale_dw = dw / input_w * seg_w;
        int scale_dh = dh / input_h * seg_h;

        cv::Rect roi(scale_dw, scale_dh, seg_w - 2 * scale_dw, seg_h - 2 * scale_dh);

        for (long unsigned int i = 0; i < indices.size(); i++)
        {
            cv::Mat dest, mask;
            cv::exp(-maskChannels[i], dest);
            dest = 1.0 / (1.0 + dest);
            dest = dest(roi);
            // std::cout << dest.size() << " " << dest.size().empty() << std::endl;
            cv::resize(dest, mask, cv::Size((int)width, (int)height), cv::INTER_LINEAR);
            objs[i].boxMask = mask(objs[i].rect) > 0.5f;
        }
    }
}