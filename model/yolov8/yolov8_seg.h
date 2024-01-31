#pragma once
#include <fstream>

#include "model/base/cv_model.h"

class YOLOv8Seg : public CvModel {
   public:
    YOLOv8Seg() = delete;
    explicit YOLOv8Seg(const std::string &model_path, const std::string framework_type, cv::Size input_size,
                       float conf_thres, float iou_thres, cv::Size seg_size, int seg_channels);
    explicit YOLOv8Seg(const std::string &yaml_file);
    ~YOLOv8Seg();

    void detect(const cv::Mat &image, std::vector<Object> &objs) override;

   protected:
    void postprocess(const std::unordered_map<std::string, IOTensor> &output, std::vector<Object> &objs);

   private:
    cv::Size m_input_size_ = {640, 640};
    cv::Size m_seg_size_ = {160, 160}; 
    int m_seg_channels_ = 32;
    float m_conf_thres_ = 0.25f;
    float m_nms_thres_ = 0.65f;
    int topk = 100;
    int strides[3] = {8, 16, 32};
    int m_grid_num_ = 8400;
    int m_num_channels_ = 38;

    PreParam pparam_;
};