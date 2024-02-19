#include "model/sam/sam.h"

SAM::SAM(const std::string& encoder_cfg, const std::string& decoder_cfg) {
    encoder_ = std::make_shared<ImageEncoder>(encoder_cfg);
    decoder_ = std::make_shared<MaskDecoder>(decoder_cfg);
}

void SAM::setImage(const cv::Mat &input_image) {
    cv::Mat mask;
    this->pparam_ = Letterbox(input_image, mask, encoder_->input_size());

    encoder_->forward(mask, features_);
}

void SAM::predict(const std::vector<cv::Point2f> &image_point_coords, const std::vector<float> &image_point_labels, cv::Mat &output_mask) {
    auto &dw = this->pparam_.dw;
    auto &dh = this->pparam_.dh;
    auto &width = this->pparam_.width;
    auto &height = this->pparam_.height;
    auto input_w = encoder_->input_size().width;
    auto input_h = encoder_->input_size().height;
    int seg_w = 256, seg_h = 256;

    int scale_dw = dw / input_w * seg_w;
    int scale_dh = dh / input_h * seg_h;

    std::vector<cv::Point2f> resize_image_point_coords;
    preprocessPoints(image_point_coords, resize_image_point_coords);

    cv::Mat low_res_mask;
    decoder_->forward(features_, resize_image_point_coords, image_point_labels, low_res_mask);

    std::vector<cv::Mat> maskChannels;
    cv::split(low_res_mask, maskChannels);

    cv::Rect roi(scale_dw, scale_dh, seg_w - 2 * scale_dw, seg_h - 2 * scale_dh);

    cv::Mat mask = maskChannels[0](roi);
    mask = mask > 0.0f;
    cv::resize(mask, output_mask, cv::Size((int)width, (int)height), cv::INTER_LINEAR);
}

void SAM::preprocessPoints(const std::vector<cv::Point2f> &input_points, std::vector<cv::Point2f> &output_points) {
    auto &dw = this->pparam_.dw;
    auto &dh = this->pparam_.dh;
    auto &ratio = this->pparam_.ratio;

    output_points.clear();
    for (const auto& point: input_points) {
        float x = point.x / ratio + dw;
        float y = point.y / ratio + dh;
        output_points.push_back(cv::Point2f(x,y));
    }
}