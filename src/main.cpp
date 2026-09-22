#include <armor_detector.h>

int main(int argc, char** argv)
{
    cv::VideoCapture capture(
        "../assets/drone_blue3.avi", cv::CAP_FFMPEG);

    cv::Mat frame;
    armor_detector::ArmorDetector detector;

    int frame_idx = 0;
    while (true) {
        capture.read(frame);

        if (frame.empty()) {
            break;
        }

        detector.detect(frame);

        cv::imshow("装甲板识别", frame);
        int c = cv::waitKey(1);
        if (c == 27) break;
    }
    cv::waitKey(0);
    return 0;
}