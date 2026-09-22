#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <utility>

namespace armor_detector
{

class LightDescriptor
{
public:
    LightDescriptor() {};
    LightDescriptor(const cv::RotatedRect& light)
    {
        m_width  = light.size.width;
        m_length = light.size.height;
        m_center = light.center;
        m_angle  = light.angle;
        m_area   = light.size.area();
    }
    const LightDescriptor& operator=(const LightDescriptor& ld)
    {
        this->m_width  = ld.m_width;
        this->m_length = ld.m_length;
        this->m_center = ld.m_center;
        this->m_angle  = ld.m_angle;
        this->m_area   = ld.m_area;
        return *this;
    }

public:
    float m_width;
    float m_length;
    cv::Point2f m_center;
    float m_angle;
    float m_area;
};

class ArmorDetector
{
public:
    ArmorDetector() = default;
    void detect(cv::Mat& frame);

private:
    
    std::vector<std::pair<cv::RotatedRect, cv::Scalar>> m_last_armors;
    int m_lost_frames = 0;
};

} 