#include "model/yolo/yolo_seg_cutoff.h"

#include <yaml-cpp/yaml.h>

static void computeDfl(float *tensor, int dfl_len, float *box) {
    for (int b = 0; b < 4; b++) {
        float exp_t[dfl_len];
        float exp_sum = 0;
        float acc_sum = 0;
        for (int i = 0; i < dfl_len; i++) {
            exp_t[i] = exp(tensor[i + b * dfl_len]);
            exp_sum += exp_t[i];
        }

        for (int i = 0; i < dfl_len; i++) {
            acc_sum += exp_t[i] / exp_sum * i;
        }
        box[b] = acc_sum;
    }
}

YOLOSegCutoff::YOLOSegCutoff(const std::string &yaml_file) {
    YAML::Node yaml_node = YAML::LoadFile(yaml_file);

    std::string model_path = yaml_node["model_path"].as<std::string>();
    framework_type_ = yaml_node["framework"].as<std::string>();
    if (framework_type_ != "RKNN") {
        std::cout << "Only RKNN is supported" << std::endl;
        exit(0);
    }
    if (!Init(model_path, framework_type_)) {
        std::cout << "Failed to init "<< framework_type_ << " framework." << std::endl;
        exit(0);
    }

    std::vector<long> input_size = yaml_node["input_size"].as<std::vector<long>>();
    m_input_size_.width = input_size.at(0);
    m_input_size_.height = input_size.at(1);
    topk_ = yaml_node["topk"].as<int>();

    std::vector<long> seg_size = yaml_node["seg_size"].as<std::vector<long>>();
    m_seg_size_.width = seg_size.at(0);
    m_seg_size_.height = seg_size.at(1);
    m_seg_channels_ = yaml_node["seg_channels"].as<int>();
    m_class_num_ = yaml_node["class_num"].as<int>();

    config_.input_len["images"] = 3 * m_input_size_.height * m_input_size_.width;
    config_.output_len["output0"] = (m_input_size_.width / strides[0]) * (m_input_size_.height / strides[0]) * 64;
    config_.output_len["output1"] =
        (m_input_size_.width / strides[0]) * (m_input_size_.height / strides[0]) * m_class_num_;
    config_.output_len["output2"] = (m_input_size_.width / strides[0]) * (m_input_size_.height / strides[0]);
    config_.output_len["output3"] = (m_input_size_.width / strides[0]) * (m_input_size_.height / strides[0]) * 32;

    config_.output_len["output4"] = (m_input_size_.width / strides[1]) * (m_input_size_.height / strides[1]) * 64;
    config_.output_len["output5"] =
        (m_input_size_.width / strides[1]) * (m_input_size_.height / strides[1]) * m_class_num_;
    config_.output_len["output6"] = (m_input_size_.width / strides[1]) * (m_input_size_.height / strides[1]);
    config_.output_len["output7"] = (m_input_size_.width / strides[1]) * (m_input_size_.height / strides[1]) * 32;

    config_.output_len["output8"] = (m_input_size_.width / strides[2]) * (m_input_size_.height / strides[2]) * 64;
    config_.output_len["output9"] =
        (m_input_size_.width / strides[2]) * (m_input_size_.height / strides[2]) * m_class_num_;
    config_.output_len["output10"] = (m_input_size_.width / strides[2]) * (m_input_size_.height / strides[2]);
    config_.output_len["output11"] = (m_input_size_.width / strides[2]) * (m_input_size_.height / strides[2]) * 32;

    config_.output_len["proto"] = m_seg_channels_ * m_seg_size_.width * m_seg_size_.height;

    config_.is_dynamic = false;
    Status status = framework_->Init(config_);
    if (status != Status::SUCCESS) {
        std::cout << "Failed to init framework" << std::endl;
        exit(0);
    }
}

YOLOSegCutoff::~YOLOSegCutoff() {}

void YOLOSegCutoff::preprocess(const cv::Mat &input_image, cv::Mat &output_image) {
    this->pparam_ = Letterbox(input_image, output_image, m_input_size_);
}

void YOLOSegCutoff::detect(const cv::Mat &image, std::vector<Object> &objs) {
    std::unordered_map<std::string, IOTensor> input, output;

    cv::Mat input_img;
    preprocess(image, input_img);

    input["images"] = IOTensor();
    input["images"].resize(input_img.total() * input_img.elemSize());
    input["images"].shape = std::vector<int64_t>{1, m_input_size_.height, m_input_size_.width, 3};
    input["images"].data_type = DataType::UINT8;
    memcpy(input["images"].data(), input_img.ptr<uint8_t>(), input_img.total() * input_img.elemSize());

    // 输出张量设置
    output["output0"] = IOTensor();
    output["output0"].resize(config_.output_len["output0"] * sizeof(uint8_t));
    output["output0"].shape =
        std::vector<int64_t>{1, 64, m_input_size_.height / strides[0], m_input_size_.width / strides[0]};
    output["output0"].data_type = DataType::INT8;

    output["output1"] = IOTensor();
    output["output1"].resize(config_.output_len["output1"] * sizeof(uint8_t));
    output["output1"].shape =
        std::vector<int64_t>{1, m_class_num_, m_input_size_.height / strides[0], m_input_size_.width / strides[0]};
    output["output1"].data_type = DataType::INT8;

    output["output2"] = IOTensor();
    output["output2"].resize(config_.output_len["output2"] * sizeof(uint8_t));
    output["output2"].shape =
        std::vector<int64_t>{1, 1, m_input_size_.height / strides[0], m_input_size_.width / strides[0]};
    output["output2"].data_type = DataType::INT8;

    output["output3"] = IOTensor();
    output["output3"].resize(config_.output_len["output3"] * sizeof(uint8_t));
    output["output3"].shape =
        std::vector<int64_t>{1, 32, m_input_size_.height / strides[0], m_input_size_.width / strides[0]};
    output["output3"].data_type = DataType::INT8;

    output["output4"] = IOTensor();
    output["output4"].resize(config_.output_len["output4"] * sizeof(uint8_t));
    output["output4"].shape =
        std::vector<int64_t>{1, 64, m_input_size_.height / strides[1], m_input_size_.width / strides[1]};
    output["output4"].data_type = DataType::INT8;

    output["output5"] = IOTensor();
    output["output5"].resize(config_.output_len["output5"] * sizeof(uint8_t));
    output["output5"].shape =
        std::vector<int64_t>{1, m_class_num_, m_input_size_.height / strides[1], m_input_size_.width / strides[1]};
    output["output5"].data_type = DataType::INT8;

    output["output6"] = IOTensor();
    output["output6"].resize(config_.output_len["output6"] * sizeof(uint8_t));
    output["output6"].shape =
        std::vector<int64_t>{1, 1, m_input_size_.height / strides[1], m_input_size_.width / strides[1]};
    output["output6"].data_type = DataType::INT8;

    output["output7"] = IOTensor();
    output["output7"].resize(config_.output_len["output7"] * sizeof(uint8_t));
    output["output7"].shape =
        std::vector<int64_t>{1, 32, m_input_size_.height / strides[1], m_input_size_.width / strides[1]};
    output["output7"].data_type = DataType::INT8;

    output["output8"] = IOTensor();
    output["output8"].resize(config_.output_len["output8"] * sizeof(uint8_t));
    output["output8"].shape =
        std::vector<int64_t>{1, 64, m_input_size_.height / strides[2], m_input_size_.width / strides[2]};
    output["output8"].data_type = DataType::INT8;

    output["output9"] = IOTensor();
    output["output9"].resize(config_.output_len["output9"] * sizeof(uint8_t));
    output["output9"].shape =
        std::vector<int64_t>{1, m_class_num_, m_input_size_.height / strides[2], m_input_size_.width / strides[2]};
    output["output9"].data_type = DataType::INT8;

    output["output10"] = IOTensor();
    output["output10"].resize(config_.output_len["output10"] * sizeof(uint8_t));
    output["output10"].shape =
        std::vector<int64_t>{1, 1, m_input_size_.height / strides[2], m_input_size_.width / strides[2]};
    output["output10"].data_type = DataType::INT8;

    output["output11"] = IOTensor();
    output["output11"].resize(config_.output_len["output11"] * sizeof(uint8_t));
    output["output11"].shape =
        std::vector<int64_t>{1, 32, m_input_size_.height / strides[2], m_input_size_.width / strides[2]};
    output["output11"].data_type = DataType::INT8;

    output["proto"] = IOTensor();
    output["proto"].resize(config_.output_len["proto"] * sizeof(uint8_t));
    output["proto"].shape = std::vector<int64_t>{1, m_seg_channels_, m_seg_size_.height, m_seg_size_.width};
    output["proto"].data_type = DataType::INT8;

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

void YOLOSegCutoff::postprocess(const std::unordered_map<std::string, IOTensor> &output, std::vector<Object> &objs) {
    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> labels;

    std::vector<float> filterSegments;
    cv::Mat protos;

    int input_w = m_input_size_.width;
    int input_h = m_input_size_.height;

    int seg_w = m_seg_size_.width;
    int seg_h = m_seg_size_.height;
    auto &dw = this->pparam_.dw;
    auto &dh = this->pparam_.dh;
    auto &width = this->pparam_.width;
    auto &height = this->pparam_.height;
    auto &ratio = this->pparam_.ratio;

    int validCount = 0;

    int dfl_len = output.at("output0").shape[1] / 4;
    int grid[3][2] = {{m_input_size_.height / strides[0], m_input_size_.width / strides[0]},
                      {m_input_size_.height / strides[1], m_input_size_.width / strides[1]},
                      {m_input_size_.height / strides[2], m_input_size_.width / strides[2]}};

    // process the outputs of rknn

    validCount += decodeBoxes(output.at("output0"), output.at("output1"), output.at("output2"), output.at("output3"),
                              grid[0][0], grid[0][1], input_h, input_w, strides[0], dfl_len, bboxes, filterSegments,
                              scores, labels, m_conf_thres_);
    validCount += decodeBoxes(output.at("output4"), output.at("output5"), output.at("output6"), output.at("output7"),
                              grid[1][0], grid[1][1], input_h, input_w, strides[1], dfl_len, bboxes, filterSegments,
                              scores, labels, m_conf_thres_);
    validCount += decodeBoxes(output.at("output8"), output.at("output9"), output.at("output10"), output.at("output11"),
                              grid[2][0], grid[2][1], input_h, input_w, strides[2], dfl_len, bboxes, filterSegments,
                              scores, labels, m_conf_thres_);
    decodeMask(output.at("proto"), protos);

    // nms
    if (validCount <= 0) {
        return;
    }
    std::vector<int> indices;

    cv::dnn::NMSBoxes(bboxes, scores, m_conf_thres_, m_nms_thres_, indices);

    int cnt = 0;
    cv::Mat masks = cv::Mat(indices.size(), m_seg_channels_, CV_32F);
    float *ptr = masks.ptr<float>();
    for (auto &i : indices) {
        if (cnt >= topk_) {
            break;
        }
        cv::Rect tmp = bboxes[i];
        Object obj;
        obj.label = labels[i];
        obj.rect = tmp;
        obj.prob = scores[i];
        objs.push_back(obj);

        for (int j = 0; j < m_seg_channels_; j++) {
            *ptr = filterSegments[i * m_seg_channels_ + j];
            ptr++;
        }
        cnt += 1;
    }

    cv::Mat matmulRes = (masks * protos).t();
    cv::Mat maskMat = matmulRes.reshape(indices.size(), {seg_w, seg_h});

    std::vector<cv::Mat> maskChannels;
    cv::split(maskMat, maskChannels);
    int scale_dw = dw / input_w * seg_w;
    int scale_dh = dh / input_h * seg_h;

    cv::Rect roi(scale_dw, scale_dh, seg_w - 2 * scale_dw, seg_h - 2 * scale_dh);

    for (long unsigned int i = 0; i < indices.size(); i++) {
        cv::Mat dest, mask;
        cv::exp(-maskChannels[i], dest);
        dest = 1.0 / (1.0 + dest);
        dest = dest(roi);
        // std::cout << dest.size() << " " << dest.size().empty() << std::endl;
        cv::resize(dest, mask, cv::Size((int)width, (int)height), cv::INTER_LINEAR);
        objs[i].boxMask = mask(objs[i].rect) > 0.5f;
    }
    return;
}

int YOLOSegCutoff::decodeBoxes(const IOTensor &output1, const IOTensor &output2, const IOTensor &output3,
                               const IOTensor &output4, int grid_h, int grid_w, int height, int width, int stride,
                               int dfl_len, std::vector<cv::Rect> &boxes, std::vector<float> &segments,
                               std::vector<float> &objProbs, std::vector<int> &classId, float threshold) {
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    int8_t *box_tensor = (int8_t *)output1.data();
    int32_t box_zp = output1.zp;
    float box_scale = output1.scale;

    int8_t *score_tensor = (int8_t *)output2.data();
    int32_t score_zp = output2.zp;
    float score_scale = output2.scale;

    int8_t *score_sum_tensor = nullptr;
    int32_t score_sum_zp = 0;
    float score_sum_scale = 1.0;
    score_sum_tensor = (int8_t *)output3.data();
    score_sum_zp = output3.zp;
    score_sum_scale = output3.scale;

    int8_t *seg_tensor = (int8_t *)output4.data();
    int32_t seg_zp = output4.zp;
    float seg_scale = output4.scale;

    int8_t score_thres_i8 = qntF32ToAffine(threshold, score_zp, score_scale);
    int8_t score_sum_thres_i8 = qntF32ToAffine(threshold, score_sum_zp, score_sum_scale);

    auto &dw = this->pparam_.dw;
    auto &dh = this->pparam_.dh;
    auto &ratio = this->pparam_.ratio;

    for (int i = 0; i < grid_h; i++) {
        for (int j = 0; j < grid_w; j++) {
            int offset = i * grid_w + j;
            int max_class_id = -1;

            int offset_seg = i * grid_w + j;
            int8_t *in_ptr_seg = seg_tensor + offset_seg;

            // for quick filtering through "score sum"
            if (score_sum_tensor != nullptr) {
                if (score_sum_tensor[offset] < score_sum_thres_i8) {
                    continue;
                }
            }

            int8_t max_score = -score_zp;
            for (int c = 0; c < m_class_num_; c++) {
                if ((score_tensor[offset] > score_thres_i8) && (score_tensor[offset] > max_score)) {
                    max_score = score_tensor[offset];
                    max_class_id = c;
                }
                offset += grid_len;
            }

            // compute box
            if (max_score > score_thres_i8) {
                for (int k = 0; k < m_seg_channels_; k++) {
                    int8_t seg_element_i8 = in_ptr_seg[(k)*grid_len] - seg_zp;
                    segments.push_back(seg_element_i8);
                }

                offset = i * grid_w + j;
                float box[4];
                float before_dfl[dfl_len * 4];
                for (int k = 0; k < dfl_len * 4; k++) {
                    before_dfl[k] = deqntAffineToF32(box_tensor[offset], box_zp, box_scale);
                    offset += grid_len;
                }
                computeDfl(before_dfl, dfl_len, box);

                float x1, y1, x2, y2, w, h;
                x1 = clamp(((-box[0] + j + 0.5) * stride - dw) * ratio, 0, this->pparam_.width);
                y1 = clamp(((-box[1] + i + 0.5) * stride - dh) * ratio, 0, this->pparam_.height);
                x2 = clamp(((box[2] + j + 0.5) * stride - dw) * ratio, 0, this->pparam_.width);
                y2 = clamp(((box[3] + i + 0.5) * stride - dh) * ratio, 0, this->pparam_.height);
                w = x2 - x1;
                h = y2 - y1;
                boxes.push_back(cv::Rect(x1, y1, w, h));

                objProbs.push_back(deqntAffineToF32(max_score, score_zp, score_scale));
                classId.push_back(max_class_id);
                validCount++;
            }
        }
    }
    return validCount;
}

void YOLOSegCutoff::decodeMask(const IOTensor &input, cv::Mat &protos) {
    int8_t *input_proto = (int8_t *)input.data();
    int32_t zp_proto = input.zp;
    protos = cv::Mat(m_seg_channels_, m_seg_size_.height * m_seg_size_.width, CV_32F);
    float *ptr = protos.ptr<float>();
    for (int i = 0; i < m_seg_channels_ * m_seg_size_.height * m_seg_size_.width; i++) {
        ptr[i] = input_proto[i] - zp_proto;
    }
}
