#include <iostream>
#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/opencv.hpp>

#include "publisher.hpp"

using namespace std::chrono_literals;


ImagePublisher::ImagePublisher() 
    : Node("image_publisher"), count_(0) {
  publisher_ = this->create_publisher<sensor_msgs::msg::Image>("Image", 10);
  timer_ = this->create_wall_timer(
    100ms, std::bind(&ImagePublisher::TimerCallback, this)
  );    
}

void ImagePublisher::TimerCallback() { 
  cv::String filename = "/home/norbert/Pictures/logo.png";
  // content, bridge, and ROS message
  cv::Mat img = cv::imread(filename, cv::IMREAD_COLOR);
  cv_bridge::CvImage img_bridge;
  sensor_msgs::msg::Image img_msg;

  //create empty header and fill it
  std_msgs::msg::Header header;
  header.stamp = this->get_clock()->now();
  header.frame_id = "map";
  // encoding: sensor_msgs::image_encodings::BGR8
  std::string encoding = "bgr8";
  img_bridge = cv_bridge::CvImage(header, encoding, img);
  // from cv_bridge to sensor_msgs::Image
  img_bridge.toImageMsg(img_msg);
  RCLCPP_INFO(this->get_logger(),"Publishing: '%i'", count_++);
  publisher_->publish(img_msg);
}