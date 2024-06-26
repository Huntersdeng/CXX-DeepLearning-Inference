#include "framework/framework.h"
#include "common/common.h"

#include "model/yolo/yolo_seg.h"
#include "model/yolo/yolo_pose.h"
#include "model/yolo/yolo.h"
#include "model/yolo/yolo_seg_cutoff.h"
#include "model/yolo/yolo_cutoff.h"

void YOLODetTest() {
    std::string current_path = "../";
    std::string yaml_file = current_path + "config/yolo/yolo.yaml";

    YOLO model(yaml_file);

    std::vector<std::string> imagePathList;
    std::string input_path = current_path + "test/image/detect"; 
    std::string output_path = current_path + "output/yolo/detect";
    cv::glob(input_path + "/*.jpg", imagePathList);

    cv::Mat image, input_image, res;
    std::vector<Object> objs;

    std::vector<std::string> class_names;
    ReadClassNames(current_path + "config/yolo/coco.txt", class_names);

    for (auto& path : imagePathList) {
        objs.clear();
        std::cout << path << std::endl;
        image = cv::imread(path);
        cv::cvtColor(image, input_image, cv::COLOR_BGR2RGB);
        model.detect(input_image, objs);
        DrawObjects(image, res, objs, class_names, COLORS);

        std::string::size_type iPos = path.find_last_of('/') + 1;
	    std::string filename = path.substr(iPos, path.length() - iPos);
        std::string out_path = output_path + "/" + filename;
        // cv::imshow("image", res);
        // cv::waitKey(0);
        cv::imwrite(out_path, res);
    }
}

void YOLODetCutoffTest() {
    std::string current_path = "../";
    std::string yaml_file = current_path + "config/yolo/yolo_cutoff.yaml";

    YOLOCutoff model(yaml_file);

    std::vector<std::string> imagePathList;
    std::string input_path = current_path + "test/image/detect"; 
    std::string output_path = current_path + "output/yolo/detect";
    cv::glob(input_path + "/*.jpg", imagePathList);

    cv::Mat image, input_image, res;
    std::vector<Object> objs;

    std::vector<std::string> class_names;
    ReadClassNames(current_path + "config/yolo/coco.txt", class_names);

    for (auto& path : imagePathList) {
        objs.clear();
        std::cout << path << std::endl;
        image = cv::imread(path);
        cv::cvtColor(image, input_image, cv::COLOR_BGR2RGB);

        auto start = std::chrono::system_clock::now();
        model.detect(input_image, objs);
        auto end = std::chrono::system_clock::now();
        std::cout << "Costs: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        
        DrawObjects(image, res, objs, class_names, COLORS);

        std::string::size_type iPos = path.find_last_of('/') + 1;
	    std::string filename = path.substr(iPos, path.length() - iPos);
        std::string out_path = output_path + "/" + filename;
        // cv::imshow("image", res);
        // cv::waitKey(0);
        cv::imwrite(out_path, res);
    }
}

void YOLOSegTest() {
    std::string current_path = "../";
    std::string yaml_file = current_path + "config/yolo/yolo_seg.yaml";

    YOLOSeg model(yaml_file);

    std::vector<std::string> imagePathList;
    std::string input_path = current_path + "test/image/detect"; 
    std::string output_path = current_path + "output/yolo/segment";
    cv::glob(input_path + "/*.jpg", imagePathList);

    cv::Mat image, input_image, res;
    std::vector<Object> objs;

    std::vector<std::string> class_names;
    ReadClassNames(current_path + "config/yolo/stardust.txt", class_names);

    for (auto& path : imagePathList) {
        objs.clear();
        std::cout << path << std::endl;
        image = cv::imread(path);
        cv::cvtColor(image, input_image, cv::COLOR_BGR2RGB);

        auto start = std::chrono::system_clock::now();
        model.detect(input_image, objs);
        auto end = std::chrono::system_clock::now();
        std::cout << "Costs: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        
        DrawObjectsMasks(image, res, objs, class_names, COLORS, MASK_COLORS);

        std::string::size_type iPos = path.find_last_of('/') + 1;
	    std::string filename = path.substr(iPos, path.length() - iPos);
        std::string out_path = output_path + "/" + filename;
        // cv::imshow("image", res);
        // cv::waitKey(0);
        cv::imwrite(out_path, res);
    }
}

void YOLOSegCutoffTest() {
    std::string current_path = "../";
    std::string yaml_file = current_path + "config/yolo/yolo_seg_cutoff.yaml";

    YOLOSegCutoff model(yaml_file);

    std::vector<std::string> imagePathList;
    std::string input_path = current_path + "test/image/detect"; 
    std::string output_path = current_path + "output/yolo/segment";
    cv::glob(input_path + "/*.jpg", imagePathList);

    cv::Mat image, input_image, res;
    std::vector<Object> objs;

    std::vector<std::string> class_names;
    ReadClassNames(current_path + "config/yolo/coco.txt", class_names);

    for (auto& path : imagePathList) {
        objs.clear();
        std::cout << path << std::endl;
        image = cv::imread(path);
        cv::cvtColor(image, input_image, cv::COLOR_BGR2RGB);

        auto start = std::chrono::system_clock::now();
        model.detect(input_image, objs);
        auto end = std::chrono::system_clock::now();
        std::cout << "Costs: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        
        DrawObjectsMasks(image, res, objs, class_names, COLORS, MASK_COLORS);

        std::string::size_type iPos = path.find_last_of('/') + 1;
	    std::string filename = path.substr(iPos, path.length() - iPos);
        std::string out_path = output_path + "/" + filename;
        // cv::imshow("image", res);
        // cv::waitKey(0);
        cv::imwrite(out_path, res);
    }
}

void YOLOPoseTest() {
    std::string current_path = "../";
    std::string yaml_file = current_path + "config/yolo/yolo_pose.yaml";

    YOLOPose model(yaml_file);

    std::vector<std::string> imagePathList;
    std::string input_path = current_path + "test/image/detect"; 
    std::string output_path = current_path + "output/yolo/pose";
    cv::glob(input_path + "/*.jpg", imagePathList);

    cv::Mat image, input_image, res;
    std::vector<Object> objs;

    std::vector<std::string> class_names;
    ReadClassNames(current_path + "config/yolo/coco.txt", class_names);

    for (auto& path : imagePathList) {
        objs.clear();
        std::cout << path << std::endl;
        image = cv::imread(path);
        cv::cvtColor(image, input_image, cv::COLOR_BGR2RGB);
        model.detect(input_image, objs);
        DrawObjectsKps(image, res, objs, SKELETON, KPS_COLORS, LIMB_COLORS);

        std::string::size_type iPos = path.find_last_of('/') + 1;
	    std::string filename = path.substr(iPos, path.length() - iPos);
        std::string out_path = output_path + "/" + filename;
        // cv::imshow("image", res);
        // cv::waitKey(0);
        cv::imwrite(out_path, res);
    }
}

int main() {
    // YOLODetTest();
    YOLOSegTest();
    // YOLOPoseTest();
    // YOLODetCutoffTest();
    // YOLOSegCutoffTest();
}
