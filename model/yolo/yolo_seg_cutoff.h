#pragma once
#include <fstream>

#include "model/base/detection_model.h"

class YOLOSegCutoff : public DetectionModel {
   public:
    YOLOSegCutoff() = delete;
    explicit YOLOSegCutoff(const std::string &yaml_file);
    ~YOLOSegCutoff();

    void detect(const cv::Mat &image, std::vector<Object> &objs) override;

   protected:
    void preprocess(const cv::Mat &input_image, cv::Mat &output_image) override;
    void postprocess(const std::unordered_map<std::string, IOTensor> &output, std::vector<Object> &objs) override;
    int decodeBoxes(const IOTensor &output1, const IOTensor &output2, const IOTensor &output3, const IOTensor &output4,
                     int grid_h, int grid_w, int height, int width, int stride, int dfl_len,
                     std::vector<cv::Rect> &boxes, std::vector<float> &segments, std::vector<float> &objProbs,
                     std::vector<int> &classId, float threshold);
    void decodeMask(const IOTensor &input, cv::Mat &protos);

   private:
    cv::Size m_input_size_ = {640, 640};
    cv::Size m_seg_size_ = {160, 160};
    int m_seg_channels_ = 32;
    int m_class_num_ = 80;
    float m_conf_thres_ = 0.25f;
    float m_nms_thres_ = 0.65f;
    int topk_ = 100;
    int strides[3] = {8, 16, 32};
    std::string framework_type_;

    PreParam pparam_;
};