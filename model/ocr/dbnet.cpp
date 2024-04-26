#include "model/ocr/dbnet.h"

#include <yaml-cpp/yaml.h>
#include "polyclipping/clipper.hpp"

static cv::RotatedRect expandBox(cv::Point2f temp[], float ratio)
{
    ClipperLib::Path path = {
        {ClipperLib::cInt(temp[0].x), ClipperLib::cInt(temp[0].y)},
        {ClipperLib::cInt(temp[1].x), ClipperLib::cInt(temp[1].y)},
        {ClipperLib::cInt(temp[2].x), ClipperLib::cInt(temp[2].y)},
        {ClipperLib::cInt(temp[3].x), ClipperLib::cInt(temp[3].y)}};
    double area = ClipperLib::Area(path);
    double distance;
    double length = 0.0;
    for (int i = 0; i < 4; i++) {
        length = length + sqrtf(powf((temp[i].x - temp[(i + 1) % 4].x), 2) +
                                powf((temp[i].y - temp[(i + 1) % 4].y), 2));
    }

    distance = area * ratio / length;

    ClipperLib::ClipperOffset offset;
    offset.AddPath(path, ClipperLib::JoinType::jtRound,
                   ClipperLib::EndType::etClosedPolygon);
    ClipperLib::Paths paths;
    offset.Execute(paths, distance);
    
    std::vector<cv::Point> contour;
    for (size_t i = 0; i < paths[0].size(); i++) {
        contour.emplace_back(paths[0][i].X, paths[0][i].Y);
    }
    offset.Clear();
    return cv::minAreaRect(contour);
}

static bool get_mini_boxes(cv::RotatedRect& rotated_rect, cv::Point2f rect[],
                    int min_size)
{

    cv::Point2f temp_rect[4];
    rotated_rect.points(temp_rect);
    for (int i = 0; i < 4; i++) {
        for (int j = i + 1; j < 4; j++) {
            if (temp_rect[i].x > temp_rect[j].x) {
                cv::Point2f temp;
                temp = temp_rect[i];
                temp_rect[i] = temp_rect[j];
                temp_rect[j] = temp;
            }
        }
    }
    int index0 = 0;
    int index1 = 1;
    int index2 = 2;
    int index3 = 3;
    if (temp_rect[1].y > temp_rect[0].y) {
        index0 = 0;
        index3 = 1;
    } else {
        index0 = 1;
        index3 = 0;
    }
    if (temp_rect[3].y > temp_rect[2].y) {
        index1 = 2;
        index2 = 3;
    } else {
        index1 = 3;
        index2 = 2;
    }   

    rect[0] = temp_rect[index0];  // Left top coordinate
    rect[1] = temp_rect[index1];  // Left bottom coordinate
    rect[2] = temp_rect[index2];  // Right bottom coordinate
    rect[3] = temp_rect[index3];  // Right top coordinate

    if (rotated_rect.size.width < min_size ||
        rotated_rect.size.height < min_size) {
        return false;
    } else {
        return true;
    }
}

static float get_box_score(float* map, cv::Point2f rect[], int width, int height,
                    float threshold)
{

    int xmin = width - 1;
    int ymin = height - 1;
    int xmax = 0;
    int ymax = 0;

    for (int j = 0; j < 4; j++) {
        if (rect[j].x < xmin) {
            xmin = rect[j].x;
        }
        if (rect[j].y < ymin) {
            ymin = rect[j].y;
        }
        if (rect[j].x > xmax) {
            xmax = rect[j].x;
        }
        if (rect[j].y > ymax) {
            ymax = rect[j].y;
        }
    }
    float sum = 0;
    int num = 0;
    for (int i = ymin; i <= ymax; i++) {
        for (int j = xmin; j <= xmax; j++) {
            if (map[i * width + j] > threshold) {
                sum = sum + map[i * width + j];
                num++;
            }
        }
    }

    return sum / num;
}

DBNet::DBNet(const std::string &yaml_file) {
    YAML::Node yaml_node = YAML::LoadFile(yaml_file);

    std::string model_path = yaml_node["model_path"].as<std::string>();
    std::string framework_type = yaml_node["framework"].as<std::string>();

    m_box_thres_ = yaml_node["box_thres"].as<float>();
    std::vector<long> max_input_size = yaml_node["max_input_size"].as<std::vector<long>>();

    if (!Init(model_path, framework_type)) exit(0);

    config_.input_len["images"] = max_input_size[0] * max_input_size[1] * max_input_size[2] * max_input_size[3];
    config_.output_len["output"] = max_input_size[0] * 2 * max_input_size[2] * max_input_size[3];
    config_.is_dynamic = true;
    Status status = framework_->Init(config_);
    if (status != Status::SUCCESS) {
        std::cout << "Failed to init framework" << std::endl;
        exit(0);
    }
}

DBNet::~DBNet()
{
    std::cout << "Destruct dbnet" << std::endl;
}

void DBNet::preprocess(const cv::Mat &input_image, cv::Mat &output_image) {
    // mean value [0.406, 0.456, 0.485] * 255
    // std value [0.225, 0.225, 0.225] * 255
    cv::Mat mask;
    this->pparam_ = paddimg(input_image, mask, 640);
    cv::dnn::blobFromImage(mask, output_image, 1 / 57.375, cv::Size(), cv::Scalar(103.53f, 116.28f, 123.675f), false, false, CV_32F);
}

void DBNet::detect(const cv::Mat &image, std::vector<Object> &objs) {
    std::unordered_map<std::string, IOTensor> input, output;

    // 输入tensor设置
    cv::Mat nchw;
    preprocess(image, nchw);

    input["images"] = IOTensor();
    input["images"].resize(nchw.total() * nchw.elemSize());
    memcpy(input["images"].data(), nchw.ptr<uint8_t>(), nchw.total() * nchw.elemSize());
    input["images"].shape = std::vector<int64_t>{1, 3, nchw.size[2], nchw.size[3]};
    input["images"].data_type = DataType::FP32;
    

    // 输出张量设置
    output["output"] = IOTensor();
    output["output"].resize(2 * nchw.size[2] * nchw.size[3] * sizeof(float));
    output["output"].shape = std::vector<int64_t>{1, 2 ,nchw.size[2] ,nchw.size[3]};
    output["output"].data_type = DataType::FP32;

    this->framework_->forward(input, output);
    postprocess(output, objs);
}

void DBNet::postprocess(const std::unordered_map<std::string, IOTensor> &output, std::vector<Object> &objs) {
    objs.clear();

    float scale = this->pparam_.ratio;

    float * const prob = (float *)output.at("output").data();
    int height = output.at("output").shape[2];
    int width = output.at("output").shape[3];

    cv::Mat map = cv::Mat::zeros(cv::Size(width, height), CV_8UC1);
    for (int h = 0; h < height; ++h) {
        uchar *ptr = map.ptr(h);
        for (int w = 0; w < width; ++w) {
            ptr[w] = (prob[h * width + w] > 0.3) ? 255 : 0;
        }
    }

    // Extracting minimum circumscribed rectangle
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarcy;
    cv::findContours(map, contours, hierarcy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Rect> boundRect(contours.size());
    std::vector<cv::RotatedRect> box(contours.size());
    cv::Point2f rect[4];
    cv::Point2f order_rect[4];

    for (size_t i = 0; i < contours.size(); i++) {
        cv::RotatedRect rotated_rect = cv::minAreaRect(cv::Mat(contours[i]));
        if (!get_mini_boxes(rotated_rect, rect, m_box_thres_)) {
            std::cout << "box too small" <<  std::endl;
            continue;
        }

        // drop low score boxes
        float score = get_box_score(prob, rect, width, height,
                                    m_score_thres_);
        if (score < m_box_thres_) {
            // std::cout << "score too low =  " << score << ", threshold = " << m_box_thres_ <<  std::endl;
            continue;
        }

        // Scaling the predict boxes depend on EXPANDRATIO
        cv::RotatedRect expandbox = expandBox(rect, m_expand_ratio_);
        expandbox.points(rect);
        if (!get_mini_boxes(expandbox, rect, m_box_min_size_ + 2)) {  
            continue;
        }

        // Restore the coordinates to the original image
        for (int k = 0; k < 4; k++) {
            order_rect[k] = rect[k];
            order_rect[k].x = int(order_rect[k].x * scale);
            order_rect[k].y = int(order_rect[k].y * scale);
        }
        
        Object obj;
        obj.label = 0;
        obj.rect = cv::Rect2i(cv::Point(order_rect[0].x,order_rect[0].y), cv::Point(order_rect[2].x,order_rect[2].y));
        objs.push_back(obj);
    }
}