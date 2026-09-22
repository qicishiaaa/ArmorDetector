#include <armor_detector.h>

#include <algorithm>
#include <cmath>
#include <vector>
#include <utility>

void armor_detector::ArmorDetector::detect(cv::Mat& frame)
{
    cv::Mat hsv, blurred;
    cv::GaussianBlur(frame, blurred, cv::Size(3, 3), 0);
    cv::cvtColor(blurred, hsv, cv::COLOR_BGR2HSV);

    // ---------- 红蓝分离 ----------
    cv::Mat red1, red2, mask_red, mask_blue;
    cv::inRange(hsv, cv::Scalar(0, 70, 75), cv::Scalar(10, 255, 255), red1);
    cv::inRange(hsv, cv::Scalar(170, 70, 65), cv::Scalar(179, 255, 255), red2);
    cv::bitwise_or(red1, red2, mask_red);

    cv::inRange(hsv, cv::Scalar(100, 80, 40), cv::Scalar(130, 255, 255), mask_blue);

    // 开运算
    cv::Mat kernel_open  = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    // 闭运算
    cv::Mat kernel_close = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 15));

    // 本帧收集到的装甲板
    std::vector<std::pair<cv::RotatedRect, cv::Scalar>> current_armors;

    auto process_mask = [&](const cv::Mat& mask, const cv::Scalar& draw_color,
                            float min_len, float max_len,
                            float min_width, float max_width,
                            float min_dis, float max_dis,
                            float white_thresh) {
        cv::Mat bin;
        mask.copyTo(bin);
        cv::medianBlur(bin, bin, 5);

        cv::morphologyEx(bin, bin, cv::MORPH_OPEN, kernel_open);
        cv::morphologyEx(bin, bin, cv::MORPH_CLOSE, kernel_close);

        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(bin, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        std::vector<armor_detector::LightDescriptor> lights;

        // ---------- 灯条提取 ----------
        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            if (area < 1) continue;                          // 20 → 5，放小灯条进来
            if (contour.size() < 4) continue;

            cv::RotatedRect rr = cv::minAreaRect(contour);

            float long_side  = std::max(rr.size.width, rr.size.height);
            float short_side = std::min(rr.size.width, rr.size.height);

            double rect_area = (double)rr.size.width * rr.size.height;
            if (rect_area < 1e-6) continue;
            float fill_ratio = (float)(area / rect_area);
            //if (fill_ratio < 0.65f) continue;   // 太弯曲 / 不规则，不是灯条
            float min_fill = (area < 30.0) ? 0.35f : 0.65f;
            if (fill_ratio < min_fill) continue;

            if (short_side < 0.3f) continue;                 // 3 → 1
            float ratio = long_side / short_side;
            if (ratio < 1.5f || ratio > 12.0f) continue;     // 收紧到极限放宽
            if (long_side < min_len || long_side > max_len) continue;  // 改成用参数

            float angle = rr.angle;
            if (rr.size.width < rr.size.height) angle += 90.0f;
            while (angle < 0.0f)    angle += 180.0f;
            while (angle >= 180.0f) angle -= 180.0f;

            armor_detector::LightDescriptor ld;
            ld.m_width  = short_side;
            ld.m_length = long_side;
            ld.m_center = rr.center;
            ld.m_angle  = angle;
            ld.m_area   = area;
            lights.push_back(ld);
        }

        // ---------- 配对 ----------
        for (size_t i = 0; i < lights.size(); ++i) {
            for (size_t j = i + 1; j < lights.size(); ++j) {
                const armor_detector::LightDescriptor& L = lights[i];
                const armor_detector::LightDescriptor& R = lights[j];

                float angle_diff = std::abs(L.m_angle - R.m_angle);
                angle_diff = std::min(angle_diff, 180.0f - angle_diff);
                if (angle_diff > 15.0f) continue;

                float len_diff_ratio = std::abs(L.m_length - R.m_length) /
                                       std::max(L.m_length, R.m_length);
                if (len_diff_ratio > 0.25f) continue;

                float mean_len   = (L.m_length + R.m_length) * 0.5f;
                float mean_width = (L.m_width  + R.m_width)  * 0.5f;
                if (mean_len < 1e-6f) continue;

                float dx = R.m_center.x - L.m_center.x;
                float dy = R.m_center.y - L.m_center.y;
                float dis = std::sqrt(dx * dx + dy * dy);
                if (dis < min_dis || dis > max_dis) continue;

                if (dis < mean_width * 1.5f) continue;
                if (dis > mean_len * 5.0f)   continue;

                float center_dist_ratio = dis / mean_len;
                if (center_dist_ratio < 1.0f || center_dist_ratio > 5.0f) continue;

                float line_angle = std::atan2(dy, dx) * 180.0f / CV_PI;
                if (line_angle < 0.0f) line_angle += 180.0f;

                float expected = L.m_angle + 90.0f;
                while (expected >= 180.0f) expected -= 180.0f;

                float line_angle_diff = std::abs(line_angle - expected);
                line_angle_diff = std::min(line_angle_diff, 180.0f - line_angle_diff);
                if (line_angle_diff > 25.0f) continue;

                cv::Point2f center(
                    (L.m_center.x + R.m_center.x) * 0.5f,
                    (L.m_center.y + R.m_center.y) * 0.5f
                );

                float armor_angle = (L.m_angle + R.m_angle) * 0.5f - 90.0f;
                while (armor_angle < 0.0f)    armor_angle += 180.0f;
                while (armor_angle >= 180.0f) armor_angle -= 180.0f;

                cv::RotatedRect armor_rect(
                    center,
                    cv::Size2f(dis, mean_len),
                    armor_angle
                );

                // ---------- 中线采样 ----------
                int roi_w = std::max(2, (int)(dis * 0.4f));
                int roi_h = std::max(2, (int)(mean_len * 0.3f));
                cv::Rect roi(
                    (int)(center.x - roi_w / 2),
                    (int)(center.y - roi_h / 2),
                    roi_w, roi_h
                );
                if (roi.x < 0 || roi.y < 0 ||
                    roi.x + roi.width  > bin.cols ||
                    roi.y + roi.height > bin.rows) {
                    continue;
                }
                cv::Mat sub = bin(roi);
                int white_pixels = cv::countNonZero(sub);
                float white_ratio = (float)white_pixels / (float)(sub.rows * sub.cols);
                if (white_ratio > white_thresh) continue;

                current_armors.push_back({armor_rect, draw_color});
            }
        }
    };

    cv::imshow("blue_mask", mask_blue);
    cv::imshow("red_mask", mask_red);

    process_mask(
        mask_red,
        cv::Scalar(0, 0, 255),
        2, 120, 1, 35, 10, 150,
        0.30f
    );
    process_mask(
        mask_blue,
        cv::Scalar(255, 0, 0),
        1, 50, 0.1, 10, 3, 60,
        0.10f
    );

    // ---------- 时序补偿 ----------
    if (!current_armors.empty()) {
        m_last_armors = current_armors;
        m_lost_frames = 0;
    } else {
        m_lost_frames++;
        if (m_lost_frames > 3) {
            m_last_armors.clear();   
        }
    }

    // ---------- 画框 ----------
    // 画本帧检测到的
    for (const auto& item : current_armors) {
        cv::Point2f v[4];
        item.first.points(v);
        for (int k = 0; k < 4; ++k) {
            cv::line(frame, v[k], v[(k + 1) % 4], item.second, 2);
        }
    }
    if (current_armors.empty() && !m_last_armors.empty()) {
        for (const auto& item : m_last_armors) {
            cv::Point2f v[4];
            item.first.points(v);
            
            cv::Scalar faded(item.second[0],
                            item.second[1],
                            item.second[2]);
            for (int k = 0; k < 4; ++k) {
                cv::line(frame, v[k], v[(k + 1) % 4], faded, 2);
            }
        }
    }
}