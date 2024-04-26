#include "model/ocr/attention.h"

#include <yaml-cpp/yaml.h>

std::string AttnModel::detect(const cv::Mat &image) {
    std::unordered_map<std::string, IOTensor> input, output;

    // 输入tensor设置
    cv::Mat nchw;
    cv::dnn::blobFromImage(image, nchw, 1 / 64.f, m_input_size_, cv::Scalar(127.5, 127.5, 127.5), false, false, CV_32F);

    input["images"] = IOTensor();
    input["images"].shape = std::vector<int64_t>{1, static_cast<int64_t>(m_input_channel_), m_input_size_.height, m_input_size_.width};
    input["images"].data_type = DataType::FP32;
    input["images"].resize(nchw.total() * nchw.elemSize());
    memcpy(input["images"].data(), nchw.ptr<uint8_t>(), nchw.total() * nchw.elemSize());

    // 输出张量设置
    output["output"] = IOTensor();
    output["output"].shape = std::vector<int64_t>{1, static_cast<int64_t>(m_output_length_)};
    output["output"].data_type = DataType::FP32;
    output["output"].resize(m_output_length_ * sizeof(float));

    this->framework_->forward(input, output);
    return postprocess(output);
}

std::string AttnModel::postprocess(const std::unordered_map<std::string, IOTensor> &output) {
    float *const outputs = (float *)output.at("output").data();
    std::string str;
    for (size_t i = 0; i < m_output_length_; i++) {
        int idx = static_cast<int>(outputs[i]);
        if (idx != 0){
            str.push_back(alphabet_[idx - 1]);
        } else {
            break;
        }
    }
    return str;
}