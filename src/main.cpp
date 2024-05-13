//
// Created by ubuntu on 3/16/23.
//
#include "chrono"
#include "opencv2/opencv.hpp"
#include "yolov8-seg.hpp"

const std::vector<std::string> CLASS_NAMES = {
    "gap"};

const std::vector<std::vector<unsigned int>> COLORS = {
    {0, 114, 189}};

const std::vector<std::vector<unsigned int>> MASK_COLORS = {
    {255, 56, 56}};

int main(int argc, char** argv)
{
    // cuda:0
    cudaSetDevice(0);

    const std::string engine_file_path{argv[1]};
    const std::string path{argv[2]};

    std::vector<std::string> imagePathList;
    bool                     isVideo{false};

    assert(argc == 3);

    auto yolov8 = new YOLOv8_seg(engine_file_path);
    yolov8->make_pipe(true);

    if (IsFile(path)) {
        std::string suffix = path.substr(path.find_last_of('.') + 1);
        if (suffix == "jpg" || suffix == "jpeg" || suffix == "png") {
            imagePathList.push_back(path);
        }
        else if (suffix == "mp4" || suffix == "avi" || suffix == "m4v" || suffix == "mpeg" || suffix == "mov"
                 || suffix == "mkv") {
            isVideo = true;
        }
        else {
            printf("suffix %s is wrong !!!\n", suffix.c_str());
            std::abort();
        }
    }
    else if (IsFolder(path)) {
        cv::glob(path + "/*.jpg", imagePathList);
    }

    cv::Mat  res, image;
    cv::Size size         = cv::Size{640, 640};
    int      topk         = 100;
    int      seg_h        = 160;
    int      seg_w        = 160;
    int      seg_channels = 32;
    float    score_thres  = 0.25f;
    float    iou_thres    = 0.65f;

    std::vector<Object> objs;

    cv::namedWindow("result", cv::WINDOW_AUTOSIZE);

    if (isVideo) {
        cv::VideoCapture cap(path);

        if (!cap.isOpened()) {
            printf("can not open %s\n", path.c_str());
            return -1;
        }
        while (cap.read(image)) {
            objs.clear();
            yolov8->copy_from_Mat(image, size);
            auto start = std::chrono::system_clock::now();
            yolov8->infer();
            auto end = std::chrono::system_clock::now();
            yolov8->postprocess(objs, score_thres, iou_thres, topk, seg_channels, seg_h, seg_w);
            yolov8->draw_objects(image, res, objs, CLASS_NAMES, COLORS, MASK_COLORS);
            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            printf("cost %2.4lf ms\n", tc);
            cv::imshow("result", res);
            if (cv::waitKey(10) == 'q') {
                break;
            }
        }
    }
    else {
        for (auto& path : imagePathList) {
            objs.clear();
            image = cv::imread(path);
            yolov8->copy_from_Mat(image, size);
            auto start = std::chrono::system_clock::now();
            yolov8->infer();
            auto end = std::chrono::system_clock::now();
            yolov8->postprocess(objs, score_thres, iou_thres, topk, seg_channels, seg_h, seg_w);
            yolov8->draw_objects(image, res, objs, CLASS_NAMES, COLORS, MASK_COLORS);
            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            printf("cost %2.4lf ms\n", tc);
            cv::imshow("result", res);
            cv::waitKey(0);
        }
    }
    cv::destroyAllWindows();
    delete yolov8;
    return 0;
}
