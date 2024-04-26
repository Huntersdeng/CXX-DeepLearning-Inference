#include "framework/framework.h"

#include "model/yolo/yolo_pose.h"
#include <yaml-cpp/yaml.h>

YOLOPose::YOLOPose(const std::string &yaml_file)
{
    YAML::Node yaml_node = YAML::LoadFile(yaml_file);

    std::string model_path = yaml_node["model_path"].as<std::string>();
    std::string framework_type = yaml_node["framework"].as<std::string>();
    
    m_conf_thres_ = yaml_node["conf_thres"].as<float>();
    m_nms_thres_ = yaml_node["nms_thres"].as<float>();

    std::vector<long> input_size = yaml_node["input_size"].as<std::vector<long>>();
    m_input_size_.width = input_size.at(0);
    m_input_size_.height = input_size.at(1);

    if (!Init(model_path, framework_type)) exit(0);

    m_grid_num_ = 0;
    for (int i = 0; i < 3; i++)
    {
        m_grid_num_ += (m_input_size_.width / strides[i]) * (m_input_size_.height / strides[i]);
    }
    config_.input_len["images"] = 3 * m_input_size_.height * m_input_size_.width;
    config_.output_len["bboxes"] = m_grid_num_ * 4;
    config_.output_len["scores"] = m_grid_num_;
    config_.output_len["kps"] = m_grid_num_ * 51;
    config_.is_dynamic = false;
    Status status = framework_->Init(config_);
    if (status != Status::SUCCESS) {
        std::cout << "Failed to init framework" << std::endl;
        exit(0);
    }
}

YOLOPose::~YOLOPose()
{
    std::cout << "Destruct yolov8" << std::endl;
}

void YOLOPose::preprocess(const cv::Mat &input_image, cv::Mat &output_image) {
    cv::Mat mask;
    this->pparam_ = Letterbox(input_image, mask, m_input_size_);
    cv::dnn::blobFromImage(mask, output_image, 1 / 255.f, cv::Size(), cv::Scalar(0, 0, 0), false, false, CV_32F);
}

void YOLOPose::detect(const cv::Mat &image, std::vector<Object> &objs)
{
    std::unordered_map<std::string, IOTensor> input, output;

    // 输入tensor设置
    cv::Mat nchw;
    preprocess(image, nchw);

    input["images"] = IOTensor();
    input["images"].resize(nchw.total() * nchw.elemSize());
    input["images"].shape = std::vector<int64_t>{1, 3, m_input_size_.height, m_input_size_.width};
    input["images"].data_type = DataType::FP32;
    memcpy(input["images"].data(), nchw.ptr<uint8_t>(), nchw.total() * nchw.elemSize());
    

    // 输出张量设置
    output["bboxes"] = IOTensor();
    output["bboxes"].shape = std::vector<int64_t>{1, m_grid_num_, 4};
    output["bboxes"].data_type = DataType::FP32;
    output["bboxes"].resize(config_.output_len["bboxes"] * sizeof(float));

    output["scores"] = IOTensor();
    output["scores"].shape = std::vector<int64_t>{1, m_grid_num_, 1};
    output["scores"].data_type = DataType::FP32;
    output["scores"].resize(config_.output_len["scores"] * sizeof(float));

    output["kps"] = IOTensor();
    output["kps"].shape = std::vector<int64_t>{1, m_grid_num_, 51};
    output["kps"].data_type = DataType::FP32;
    output["kps"].resize(config_.output_len["kps"] * sizeof(float));

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

void YOLOPose::postprocess(const std::unordered_map<std::string, IOTensor> &output, std::vector<Object> &objs)
{
    objs.clear();
    auto num_anchors = m_grid_num_;

    auto &dw = this->pparam_.dw;
    auto &dh = this->pparam_.dh;
    auto &width = this->pparam_.width;
    auto &height = this->pparam_.height;
    auto &ratio = this->pparam_.ratio;

    float *bbox_ptr = (float *)output.at("bboxes").data();
    float *score_ptr = (float *)output.at("scores").data();
    float *kps_ptr = (float *)output.at("kps").data();

    std::vector<cv::Rect>           bboxes;
    std::vector<float>              scores;
    std::vector<int>                labels;
    std::vector<int>                indices;
    std::vector<std::vector<float>> kpss;

    for (int i = 0; i < num_anchors; i++)
    {
        float score = *(score_ptr++);
        if (score > m_conf_thres_)
        {
            float x0 = *bbox_ptr++ - dw;
            float y0 = *bbox_ptr++ - dh;
            float x1 = *bbox_ptr++ - dw;
            float y1 = *bbox_ptr++ - dh;

            x0 = clamp(x0 * ratio, 0.f, width);
            y0 = clamp(y0 * ratio, 0.f, height);
            x1 = clamp(x1 * ratio, 0.f, width);
            y1 = clamp(y1 * ratio, 0.f, height);

            std::vector<float> kps;
            for (int k = 0; k < 17; k++) {
                float kps_x = (*(kps_ptr + 3 * k) - dw) * ratio;
                float kps_y = (*(kps_ptr + 3 * k + 1) - dh) * ratio;
                float kps_s = *(kps_ptr + 3 * k + 2);
                kps_x       = clamp(kps_x, 0.f, width);
                kps_y       = clamp(kps_y, 0.f, height);
                kps.push_back(kps_x);
                kps.push_back(kps_y);
                kps.push_back(kps_s);
            }
            kps_ptr += 51;

            labels.push_back(0);
            scores.push_back(score);
            bboxes.push_back(cv::Rect_<float>(x0, y0, x1 - x0, y1 - y0));
            kpss.push_back(kps);
        } else {
            bbox_ptr += 4;
            kps_ptr += 51;
        }
    }
    cv::dnn::NMSBoxes(bboxes, scores, m_conf_thres_, m_nms_thres_, indices);

    int cnt = 0;
    for (auto& i : indices) {
        if (cnt >= topk) {
            break;
        }
        Object obj;
        obj.rect  = bboxes[i];
        obj.prob  = scores[i];
        obj.label = labels[i];
        obj.kps   = kpss[i];
        objs.push_back(obj);
        cnt += 1;
    }
}