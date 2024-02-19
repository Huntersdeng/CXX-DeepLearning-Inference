#include "framework/framework.h"
#include "common/common.h"

#include "model/sam/sam.h"

int main() {
    std::string current_path = "../";
    std::string encoder_cfg = current_path + "config/sam/image_encoder.yaml";
    std::string decoder_cfg = current_path + "config/sam/mask_decoder.yaml";

    SAM sam(encoder_cfg, decoder_cfg);

    cv::Mat image, input_image;

    image = cv::imread("../test/image/sam/dogs.jpg");
    cv::cvtColor(image, input_image, cv::COLOR_BGR2RGB);

    sam.setImage(input_image);

    std::vector<cv::Point2f> points;
    points.push_back(cv::Point2f(100, 100));
    points.push_back(cv::Point2f(850, 759));
    std::vector<float> labels{2, 3};

    cv::Mat output_mask;
    sam.predict(points, labels, output_mask);

    cv::Mat res = image.clone();
    cv::Mat mask = image.clone();

    // cv::rectangle(res, cv::Rect(100, 100, 750, 659), {0, 0, 255}, -1);
    // mask.setTo(cv::Scalar(255, 56, 56), output_mask);
    // cv::addWeighted(res, 0.5, mask, 0.8, 1, res);
    cv::imwrite("../output/sam/dogs.jpg", output_mask);
}