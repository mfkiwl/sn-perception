#pragma once
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

/* 
A simple 'cv_bridge' based image publisher node for ROS2.
 */
class ImagePublisher : public rclcpp::Node {
 public:
  ImagePublisher();

 private:
  void TimerCallback();
    
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
  size_t count_;
};