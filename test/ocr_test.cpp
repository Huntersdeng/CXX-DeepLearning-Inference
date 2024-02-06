#include "framework/framework.h"
#include "common/common.h"

#include "model/ocr/ctc.h"
#include "model/ocr/attention.h"

void CtcModelTest() {
    std::string current_path = "../";
    std::string yaml_file = current_path + "config/ocr/ctc.yaml";

    CtcModel model(yaml_file);

    std::vector<std::string> imagePathList;
    std::string input_path = current_path + "test/image/ocr"; 
    cv::glob(input_path + "/*.png", imagePathList);

    cv::Mat image, res;

    for (auto& path : imagePathList) {
        auto start = std::chrono::system_clock::now();
        image = cv::imread(path);
        std::string output = model.detect(image);
        std::cout << path << ": " << output << std::endl;
        auto end = std::chrono::system_clock::now();
        auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
        printf("cost %2.4lf ms\n", tc);
    }
}

void AttnModelTest() {
    std::string current_path = "../";
    std::string yaml_file = current_path + "config/ocr/attn.yaml";

    AttnModel model(yaml_file);

    std::vector<std::string> imagePathList;
    std::string input_path = current_path + "test/image/ocr"; 
    cv::glob(input_path + "/*.png", imagePathList);

    cv::Mat image, res;

    for (auto& path : imagePathList) {
        auto start = std::chrono::system_clock::now();
        image = cv::imread(path);
        std::string output = model.detect(image);
        std::cout << path << ": " << output << std::endl;
        auto end = std::chrono::system_clock::now();
        auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
        printf("cost %2.4lf ms\n", tc);
    }
}

int main() {
    CtcModelTest();
    // AttnModelTest();
}