#pragma once
#include "model/sam/image_encoder.h"
#include "model/sam/mask_decoder.h"

namespace sam {

class SAM {
   public:
    SAM() = delete;
    SAM(const std::string &encoder_cfg, const std::string &decoder_cfg);
    ~SAM(){};
    void setImage(const cv::Mat &image);
    void predict(const std::vector<cv::Point2f> &image_point_coords, const std::vector<float> &image_point_labels,
                 cv::Mat &output_mask);
    void preprocessPoints(const std::vector<cv::Point2f> &input_points, std::vector<cv::Point2f> &output_points);

   private:
    std::shared_ptr<ImageEncoder> encoder_;
    std::shared_ptr<MaskDecoder> decoder_;
    PreParam pparam_;
    IOTensor features_;
};
}  // namespace sam