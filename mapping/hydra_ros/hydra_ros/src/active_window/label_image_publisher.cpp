/* -----------------------------------------------------------------------------
 * Copyright 2022 Massachusetts Institute of Technology.
 * All Rights Reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Research was sponsored by the United States Air Force Research Laboratory and
 * the United States Air Force Artificial Intelligence Accelerator and was
 * accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views
 * and conclusions contained in this document are those of the authors and should
 * not be interpreted as representing the official policies, either expressed or
 * implied, of the United States Air Force or the U.S. Government. The U.S.
 * Government is authorized to reproduce and distribute reprints for Government
 * purposes notwithstanding any copyright notation herein.
 * -------------------------------------------------------------------------- */
#include "hydra_ros/active_window/label_image_publisher.h"

#include <config_utilities/config.h>
#include <config_utilities/factory.h>
#include <config_utilities/printing.h>
#include <config_utilities/validation.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

namespace hydra {

void declare_config(LabelImagePublisher::Config& config) {
  using namespace config;
  name("LabelImagePublisher::Config");
  field(config.ns, "ns");
}

LabelImagePublisher::LabelImagePublisher(const Config& config)
    : config(config::checkValid(config)) 
    {
        nh_ = ros::NodeHandle(config.ns);
        publisher_ = nh_.advertise<sensor_msgs::Image>("label_image", 1);
    }

LabelImagePublisher::~LabelImagePublisher() {
    // Your cleanup code here, if any
}

void LabelImagePublisher::call(uint64_t timestamp_ns,
                                  const VolumetricMap& map,
                                  ActiveWindowOutput& output) const {
  publish(timestamp_ns, output);
}

void LabelImagePublisher::publish(uint64_t timestamp_ns, const ActiveWindowOutput data) const {
    
    cv::Mat label_image_16u = data.sensor_data->label_image;
    const std::vector<DetectionFeature>& features = data.sensor_data->features;

    double min_val_double, max_val_double;
    cv::minMaxLoc(label_image_16u, &min_val_double, &max_val_double);
    int max_label_id = static_cast<int>(max_val_double);

    cv::Mat color_image = cv::Mat::zeros(label_image_16u.rows, label_image_16u.cols, CV_8UC3);
    std::vector<cv::Vec3b> label_colors_bgr;

    if (max_label_id >= 0) { // Handle case where image might be all zeros
        label_colors_bgr.push_back(cv::Vec3b(0, 0, 0)); // Color for label 0 (background)

        if (max_label_id > 0) {
            for (int i = 1; i <= max_label_id; ++i) {
                uchar hue = static_cast<uchar>((( (float)i / (float)(max_label_id + 1.0f) ) * 179.0f));
                cv::Mat hsv_color_mat(1, 1, CV_8UC3, cv::Scalar(hue, 255, 220));
                cv::Mat bgr_color_mat;
                cv::cvtColor(hsv_color_mat, bgr_color_mat, cv::COLOR_HSV2BGR);
                label_colors_bgr.push_back(bgr_color_mat.at<cv::Vec3b>(0, 0));
            }
        }

        // Apply colors
        for (int r = 0; r < label_image_16u.rows; ++r) {
            const uint16_t* row_ptr_label = label_image_16u.ptr<uint16_t>(r);
            cv::Vec3b* row_ptr_color = color_image.ptr<cv::Vec3b>(r);
            for (int c = 0; c < label_image_16u.cols; ++c) {
                uint16_t label_id = row_ptr_label[c];
                if (label_id < label_colors_bgr.size()) {
                    row_ptr_color[c] = label_colors_bgr[label_id];
                } else {
                    row_ptr_color[c] = label_colors_bgr[0]; // Default to background
                }
            }
        }

        // Draw text for each label (object)
        if (max_label_id > 0) {
            for (int current_label_id = 1; current_label_id <= max_label_id; ++current_label_id) {
                // Get the object name. Adapt this line to your DetectionFeature structure.
                const DetectionFeature& feature = features[current_label_id - 1];
                std::string object_text = feature.class_name;
                // If using a getter: std::string object_text = feature.getDisplayName();


                // Find a position for the text (centroid of the mask for this label)
                cv::Mat single_label_mask;
                cv::compare(label_image_16u, static_cast<uint16_t>(current_label_id), single_label_mask, cv::CMP_EQ);

                if (cv::countNonZero(single_label_mask) > 0) {
                    cv::Moments m = cv::moments(single_label_mask, true); // true for binary image (mask)
                    if (m.m00 > 0) { // Ensure area is not zero
                        cv::Point centroid(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));

                        // Text properties
                        int font_face = cv::FONT_HERSHEY_SIMPLEX;
                        double font_scale = 0.4; // Adjust based on image size and desired text size
                        int thickness = 1;
                        cv::Scalar text_color_main(255, 255, 255); // White
                        cv::Scalar text_color_outline(0, 0, 0);   // Black

                        // Make sure centroid is valid for drawing
                        if(centroid.x > 0 && centroid.y > 0 && centroid.x < color_image.cols && centroid.y < color_image.rows) {
                            // Draw outline (draw thicker black text first)
                            cv::putText(color_image, object_text, centroid, font_face, font_scale, text_color_outline, thickness + 1, cv::LINE_AA);
                            // Draw main text
                            cv::putText(color_image, object_text, centroid, font_face, font_scale, text_color_main, thickness, cv::LINE_AA);
                        }
                    }
                }
            }
        }
    } else {
        ROS_INFO_THROTTLE(1.0, "ColorizeAndDrawNames: No labels > 0 found in image.");
    }
    // Convert to sensor_msgs::Image
    sensor_msgs::Image msg;
    std_msgs::Header header_to_use;
    msg.header.stamp.fromNSec(timestamp_ns);
    cv_bridge::CvImage cv_image_out(header_to_use, sensor_msgs::image_encodings::RGB8, color_image);
    cv_image_out.toImageMsg(msg);
    // Publish the message
    publisher_.publish(msg);
}

namespace {

static const auto registration_ =
    config::RegistrationWithConfig<ReconstructionModule::Sink,
                                   LabelImagePublisher,
                                   LabelImagePublisher::Config>(
        "LabelImagePublisher");

}
}  // namespace hydra
