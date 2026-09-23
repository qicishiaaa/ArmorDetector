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
    cv::inRange(hsv, cv::Scalar(0, 60, 80), cv::Scalar(12, 255, 255), red1);
    cv::inRange(hsv, cv::Scalar(165, 60, 80), cv::Scalar(179, 255, 255), red2);
    cv::bitwise_or(red1, red2, mask_red);

    cv::Mat blue_shallow, blue_deep;
    // 浅蓝段：H 低，S 低也收
    cv::inRange(hsv, cv::Scalar(85, 60, 70), cv::Scalar(105, 255, 255), blue_shallow);
    // 深蓝段：H 高，S 要求高
    cv::inRange(hsv, cv::Scalar(110, 150, 150), cv::Scalar(128, 255, 255), blue_deep);
    cv::bitwise_or(blue_shallow, blue_deep, mask_blue);

    // 闭运算
    cv::Mat kernel_close = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 7));

    // 本帧收集到的装甲板
    std::vector<std::pair<cv::RotatedRect, cv::Scalar>> current_armors;

    auto process_mask = [&](const cv::Mat& mask, const cv::Scalar& draw_color,
                            float min_len, float max_len,
                            float min_width, float max_width,
                            float min_dis, float max_dis,
                            float white_thresh) {
        cv::Mat bin;
        mask.copyTo(bin);
        cv::medianBlur(bin, bin, 3);
        
        cv::Mat kernel_close_used = kernel_close;
        if (min_len < 10.0f) {
            kernel_close_used = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 5));
        }
        cv::morphologyEx(bin, bin, cv::MORPH_CLOSE, kernel_close_used);

        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(bin, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        std::vector<armor_detector::LightDescriptor> lights;

        // ---------- 灯条提取 ----------
        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            if (area < 1) continue;  
            if (contour.size() < 4) continue;

            cv::RotatedRect rr = cv::minAreaRect(contour);

            // 用轮廓质心作为灯条中心
            cv::Moments mu = cv::moments(contour);
            if (mu.m00 <= 1e-6) continue;
            cv::Point2f light_center(
                static_cast<float>(mu.m10 / mu.m00),
                static_cast<float>(mu.m01 / mu.m00)
            );

            float long_side  = std::max(rr.size.width, rr.size.height);
            float short_side = std::min(rr.size.width, rr.size.height);

            double rect_area = (double)rr.size.width * rr.size.height;
            if (rect_area < 1e-6) continue;
            float fill_ratio = (float)(area / rect_area);
            //if (fill_ratio < 0.65f) continue;  
            float min_fill = (area < 10.0) ? 0.35f : 0.60f;
            if (fill_ratio < min_fill) continue;

            if (short_side < 0.3f) continue;                 
            float ratio = long_side / short_side;
            float min_ratio = (long_side < 10.0f) ? 1.2f : 1.5f; //add
            if (ratio < min_ratio || ratio > 8.0f) continue;
            //if (ratio < 1.5f || ratio > 12.0f) continue;     
            if (long_side < min_len || long_side > max_len) continue;  

            float angle = rr.angle;
            if (rr.size.width < rr.size.height) angle += 90.0f;
            while (angle < 0.0f)    angle += 180.0f;
            while (angle >= 180.0f) angle -= 180.0f;

            armor_detector::LightDescriptor ld;
            ld.m_width  = short_side;
            ld.m_length = long_side;
            //ld.m_center = rr.center;
            ld.m_center = light_center;
            
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
                
                float mean_len   = (L.m_length + R.m_length) * 0.5f;
                float mean_width = (L.m_width  + R.m_width)  * 0.5f;
                if (mean_len < 1e-6f) continue;

                float max_len_diff = (mean_len < 12.0f) ? 0.45f : 0.25f;
                if (len_diff_ratio > max_len_diff) continue;

                float dx = R.m_center.x - L.m_center.x;
                float dy = R.m_center.y - L.m_center.y;
                float dis = std::sqrt(dx * dx + dy * dy);
                if (dis < min_dis || dis > max_dis) continue;

                if (dis < mean_width * 1.2f) continue;
                if (dis > mean_len * 5.0f)   continue;

                float center_dist_ratio = dis / mean_len;
                
                float min_cdr = 1.0f;
                float max_cdr = (mean_len < 12.0f) ? 5.0f : 4.0f;
                if (center_dist_ratio < min_cdr || center_dist_ratio > max_cdr) continue;

                float line_angle = std::atan2(dy, dx) * 180.0f / CV_PI;
                if (line_angle < 0.0f) line_angle += 180.0f;

                float expected = L.m_angle + 90.0f;
                while (expected >= 180.0f) expected -= 180.0f;

                float line_angle_diff = std::abs(line_angle - expected);
                line_angle_diff = std::min(line_angle_diff, 180.0f - line_angle_diff);
                
                float max_line_angle_diff;
                
                if (mean_len < 8.0f)       max_line_angle_diff = 50.0f;
                else if (mean_len < 12.0f) max_line_angle_diff = 35.0f;
                else                       max_line_angle_diff = 25.0f;
                if (line_angle_diff > max_line_angle_diff) continue;

                cv::Point2f center(
                    (L.m_center.x + R.m_center.x) * 0.5f,
                    (L.m_center.y + R.m_center.y) * 0.5f
                );

                // 宽度方向直接用两灯条中心连线，避免灯条角度平均带来的抖动
                float armor_angle = line_angle;

                // 宽度用“中心距 + 平均灯条宽度”
                float armor_width  = dis + mean_width;
                float armor_height = mean_len;

                cv::RotatedRect armor_rect(
                    center,
                    cv::Size2f(armor_width, armor_height),
                    armor_angle
                );

                cv::RotatedRect inner = armor_rect;
                inner.size.width  *= 0.5f;
                inner.size.height *= 0.5f;

                cv::Point2f inner_pts[4];
                inner.points(inner_pts);

                std::vector<cv::Point> poly;
                poly.reserve(4);
                for (int k = 0; k < 4; ++k) {
                    poly.emplace_back(cv::Point(cvRound(inner_pts[k].x), cvRound(inner_pts[k].y)));
                }

                cv::Mat roi_mask = cv::Mat::zeros(bin.size(), CV_8UC1);
                cv::fillConvexPoly(roi_mask, poly, 255);

                int roi_area = cv::countNonZero(roi_mask);
                if (roi_area <= 0) continue;

                cv::Mat sub;
                cv::bitwise_and(bin, roi_mask, sub);
                int white_pixels = cv::countNonZero(sub);
                float white_ratio = (float)white_pixels / (float)roi_area;

                float thresh_used = (mean_len < 10.0f) ? 0.15f : white_thresh;
                if (white_ratio > thresh_used) continue;

                current_armors.push_back({armor_rect, draw_color});
            }
        }
    };

    process_mask(
        mask_red,
        cv::Scalar(0, 0, 255),
        3, 120, 1, 35, 10, 150,
        0.30f
    );
 
    // 浅蓝
    process_mask(
        blue_shallow,
        cv::Scalar(255, 0, 0),
        4, 70, 1, 10, 6, 60,
        0.20f
    );
    // 深蓝
    process_mask(
        blue_deep,
        cv::Scalar(255, 0, 0),
        10, 70, 3, 15, 20, 90,
        0.15f
    );

    //---------- 时序补偿 ----------
    if (!current_armors.empty()) {
        m_last_armors = current_armors;
        m_lost_frames = 0;
    } else {
        m_lost_frames++;
        if (m_lost_frames > 5) {
            m_last_armors.clear();   
        }
    }

    // ---------- 时序平滑（EMA）----------
    std::vector<std::pair<cv::RotatedRect, cv::Scalar>> smoothed;
    for (const auto& cur : current_armors) {
        bool matched = false;
        for (const auto& prev : m_smoothed_armors) {
            float dx = cur.first.center.x - prev.first.center.x;
            float dy = cur.first.center.y - prev.first.center.y;
            if (std::sqrt(dx*dx + dy*dy) < 30.0f) {
                // 位置和尺寸用本帧的
                float new_cx = cur.first.center.x;
                float new_cy = cur.first.center.y;
                float new_w  = cur.first.size.width;
                float new_h  = cur.first.size.height;

                // 只有角度做 EMA 平滑
                float a_diff = cur.first.angle - prev.first.angle;
                while (a_diff > 90.0f)  a_diff -= 180.0f;
                while (a_diff < -90.0f) a_diff += 180.0f;
                float new_angle = prev.first.angle + a_diff * 0.7f;

                cv::RotatedRect smoothed_rect(
                    cv::Point2f(new_cx, new_cy),
                    cv::Size2f(new_w, new_h),
                    new_angle
                );
                smoothed.push_back({smoothed_rect, cur.second});
                matched = true;
                break;
            }
        }
        if (!matched) {
            smoothed.push_back(cur);
        }
    }
    m_smoothed_armors = smoothed;
    current_armors = smoothed;

    // ---------- 画框 ----------
    // 画本帧检测到的
    for (const auto& item : current_armors) {
        cv::Point2f v[4];
        item.first.points(v);
        for (int k = 0; k < 4; ++k) {
            cv::line(frame, v[k], v[(k + 1) % 4], item.second, 2);
        }
    }
}