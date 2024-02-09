#include "framework/framework.h"
#include "common/common.h"

#include "model/ocr/ctc.h"
#include "model/ocr/attention.h"
#include "model/ocr/dbnet.h"

void CtcModelTest() {
    std::string current_path = "../";
    std::string yaml_file = current_path + "config/ocr/rec/ctc.yaml";

    CtcModel model(yaml_file);

    std::vector<std::string> imagePathList;
    std::string input_path = current_path + "test/image/ocr/rec"; 
    cv::glob(input_path + "/*.png", imagePathList);

    cv::Mat image, res;

    for (auto& path : imagePathList) {
        auto start = std::chrono::system_clock::now();
        image = cv::imread(path, 0);
        std::string output = model.detect(image);
        std::cout << path << ": " << output << std::endl;
        auto end = std::chrono::system_clock::now();
        auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
        printf("cost %2.4lf ms\n", tc);
    }
}

void AttnModelTest() {
    std::string current_path = "../";
    std::string yaml_file = current_path + "config/ocr/rec/attn.yaml";

    AttnModel model(yaml_file);

    std::vector<std::string> imagePathList;
    std::string input_path = current_path + "test/image/ocr/rec"; 
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

void DBNetTest() {
    std::string current_path = "../";
    std::string yaml_file = current_path + "config/ocr/det/dbnet.yaml";

    DBNet model(yaml_file);

    std::vector<std::string> imagePathList;
    std::string input_path = current_path + "test/image/ocr/det"; 
    std::string output_path = current_path + "output/dbnet";
    cv::glob(input_path + "/*.png", imagePathList);

    cv::Mat image, input_image, res;
    std::vector<Object> objs;

    for (auto& path : imagePathList) {
        objs.clear();
        std::cout << path << std::endl;
        image = cv::imread(path);
        cv::cvtColor(image, input_image, cv::COLOR_BGR2RGB);
        model.detect(input_image, objs);
        DrawBoxes(image, res, objs);

        std::string::size_type iPos = path.find_last_of('/') + 1;
	    std::string filename = path.substr(iPos, path.length() - iPos);
        std::string out_path = output_path + "/" + filename;
        // cv::imshow("image", res);
        // cv::waitKey(0);
        cv::imwrite(out_path, res);
    }
}

int main() {
    // CtcModelTest();
    // AttnModelTest();
    DBNetTest();
}