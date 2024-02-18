#include "model/sam/sam.h"

SAM::SAM(const std::string& encoder_cfg, const std::string& decoder_cfg) {
    encoder_ = std::make_shared<ImageEncoder>(encoder_cfg);
    decoder_ = std::make_shared<MaskDecoder>(decoder_cfg);
}

void SAM::set_image(const cv::Mat &input_image) {
    cv::Mat mask;
    this->pparam_ = Letterbox(input_image, mask, encoder_->input_size());

    encoder_->forward(mask, features_);
}

void SAM::predict(const std::vector<cv::Point2i> &image_point_coords, const std::vector<float> &image_point_labels, cv::Mat &output_mask) {
    cv::Mat low_res_mask;
    decoder_->forward(features_, image_point_coords, image_point_labels, low_res_mask);

    std::vector<cv::Mat> maskChannels;
    cv::split(low_res_mask, maskChannels);

    auto &dw = this->pparam_.dw;
    auto &dh = this->pparam_.dh;
    auto &width = this->pparam_.width;
    auto &height = this->pparam_.height;
    auto input_w = encoder_->input_size().width;
    auto input_h = encoder_->input_size().height;
    int seg_w = 256, seg_h = 256;

    int scale_dw = dw / input_w * seg_w;
    int scale_dh = dh / input_h * seg_h;

    cv::Rect roi(scale_dw, scale_dh, seg_w - 2 * scale_dw, seg_h - 2 * scale_dh);

    cv::Mat mask = maskChannels[0](roi);
    mask = mask > 0.0f;
    cv::resize(mask, output_mask, cv::Size((int)width, (int)height), cv::INTER_LINEAR);
}