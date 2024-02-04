#include "model/ocr/ctc.h"

std::string CtcModel::detect(const cv::Mat &image) {
    std::unordered_map<std::string, IOTensor> input, output;

    // 输入tensor设置
    cv::Mat nchw;
    cv::dnn::blobFromImage(image, nchw, 1 / 127.5f, m_input_size_, cv::Scalar(127.5, 127.5, 127.5), true, false, CV_32F);

    input["images"] = IOTensor();
    input["images"].resize(nchw.total() * nchw.elemSize());
    memcpy(input["images"].data(), nchw.ptr<uint8_t>(), nchw.total() * nchw.elemSize());

    // 输出张量设置
    output["output"] = IOTensor();
    output["output"].resize(config_.output_len["output"] * sizeof(float));

    this->framework_->forward(input, output);
    return postprocess(output);
}

std::string CtcModel::postprocess(const std::unordered_map<std::string, IOTensor> &output) {
    float *const outputs = (float *)output.at("output").data();
    std::string str;
    for (size_t i = 0; i < m_output_length_; i++) {
        int idx = static_cast<int>(outputs[i]);
        if (idx == 0 || (i > 0 && static_cast<int>(outputs[i-1]) == idx)) continue;
        str.push_back(alphabet_[idx - 1]);
    }
    return str;
}