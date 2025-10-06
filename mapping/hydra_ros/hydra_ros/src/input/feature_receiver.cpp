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
#include "hydra_ros/input/feature_receiver.h"

#include <config_utilities/config.h>
#include <config_utilities/printing.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <hydra/common/global_info.h>
#include <hydra/common/pipeline_queues.h>
#include <hydra/frontend/view_selector.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

#include "hydra_ros_build_config.h"

#if defined(HYDRA_USE_SEMANTIC_INFERENCE) && HYDRA_USE_SEMANTIC_INFERENCE
#include <semantic_inference_msgs/FeatureVectorStamped.h>
#endif

namespace hydra {

void declare_config(FeatureReceiver::Config& config) {
  using namespace config;
  name("FeatureReceiver::Config");
  field(config.ns, "ns");
  field(config.queue_size, "queue_size");
  field(config.tf_lookup, "tf_lookup");
  field(config.sensors_to_exclude, "sensors_to_exclude");
  field(config.occlusion_check, "occlusion_check");
}

#if defined(HYDRA_USE_SEMANTIC_INFERENCE) && HYDRA_USE_SEMANTIC_INFERENCE
using semantic_inference_msgs::FeatureVectorStamped;

struct SynchronizedSubscriber {
  using Ptr = std::unique_ptr<SynchronizedSubscriber>;
  using Callback = std::function<PoseStatus(uint64_t)>;

  // NEW: Define message filter subscribers and the synchronizer policy
  using FeatureSubscriber = message_filters::Subscriber<FeatureVectorStamped>;
  using DepthSubscriber = message_filters::Subscriber<sensor_msgs::Image>;
  using SyncPolicy =
      message_filters::sync_policies::ApproximateTime<FeatureVectorStamped, sensor_msgs::Image>;
  using Synchronizer = message_filters::Synchronizer<SyncPolicy>;

  SynchronizedSubscriber(ros::NodeHandle& nh,
                         const std::string& sensor_name,
                         MessageQueue<SyncedMessage>::Ptr& queue,
                         size_t queue_size = 10);
  void callback(const FeatureVectorStamped::ConstPtr& feature_msg,
                const sensor_msgs::Image::ConstPtr& depth_msg);

  const std::string sensor_name;

  FeatureSubscriber feature_sub;
  DepthSubscriber depth_sub;
  std::unique_ptr<Synchronizer> sync;
  MessageQueue<SyncedMessage>::Ptr raw_message_queue_;
};

SynchronizedSubscriber::SynchronizedSubscriber(ros::NodeHandle& nh,
                                               const std::string& sensor_name,
                                               MessageQueue<SyncedMessage>::Ptr& queue,
                                               size_t queue_size)
    : sensor_name(sensor_name), raw_message_queue_(queue) {
  const std::string feature_topic = sensor_name + "/feature";
  const std::string depth_topic = sensor_name + "/depth_registered/image_rect";
  feature_sub.subscribe(nh, feature_topic, queue_size);
  depth_sub.subscribe(nh, depth_topic, queue_size);

  sync = std::make_unique<Synchronizer>(SyncPolicy(queue_size), feature_sub, depth_sub);
  sync->registerCallback(boost::bind(&SynchronizedSubscriber::callback, this, _1, _2));
  
}

void SynchronizedSubscriber::callback(const FeatureVectorStamped::ConstPtr& feature_msg,
                                      const sensor_msgs::Image::ConstPtr& depth_msg) {

    raw_message_queue_->push({feature_msg, depth_msg, sensor_name});
}

void FeatureReceiver::start() {
  std::set<std::string> to_exclude(config.sensors_to_exclude.begin(),
                                   config.sensors_to_exclude.end());
  const auto sensor_names = GlobalInfo::instance().getAvailableSensors();
  for (const auto& name : sensor_names) {
    if (to_exclude.count(name)) {
      continue;
    }

    subs_.push_back(std::make_unique<SynchronizedSubscriber>(
        nh_,
        name,
        raw_message_queue_,
        config.queue_size));
  }
  processing_thread_ = std::thread(&FeatureReceiver::processingQueue, this);
}
#else

struct FeatureSubscriber {};

void FeatureReceiver::start() {
  LOG(ERROR) << "semantic_inference_msgs not found when building, disabled!";
}
#endif

void FeatureReceiver::processingQueue() {
  while (!should_shutdown_) {
    // Use poll() to efficiently wait for new messages
    const bool has_data = raw_message_queue_->poll();
    if (!has_data) {
      continue;
    }
    // Pop the message and process it
    SyncedMessage packet = raw_message_queue_->pop();

    const auto timestamp_ns = packet.features->header.stamp.toNSec();
    const auto& vec = packet.features->feature.data;

    const auto sensor = GlobalInfo::instance().getSensor(packet.sensor_name);
    auto pose_callback = [this](uint64_t timestamp_ns) {
      return lookup_.getBodyPose(timestamp_ns);
    };
    const auto pose_status = pose_callback(timestamp_ns);
    if (!pose_status) {
      LOG(WARNING) << "Dropping synchronized packet @ " << timestamp_ns
                  << "[ns] for sensor " << packet.sensor_name << " due to missing pose.";
      continue;
    }

    const Eigen::Isometry3d world_T_sensor =
        pose_status.target_T_source() * sensor->body_T_sensor();

    cv::Mat depth_image = cv_bridge::toCvCopy(packet.depth)->image;
    cv::Mat depth_converted;
    if (depth_image.type() == CV_16UC1)
    {
      depth_image.convertTo(depth_converted, CV_32FC1, 1.0e-3);
    }
    else if (depth_image.type() != CV_32FC1)
    {
      LOG(ERROR) << "Unsupported depth image type: " << depth_image.type();
      continue;
    }
    else
    {
      depth_converted = depth_image;
    }

    // If we need to do occlusion checking, use the constructor that takes in the depth image
    std::unique_ptr<FeatureView> output_packet;
    if (config.occlusion_check)
    {
      output_packet = std::make_unique<FeatureView>(
          timestamp_ns,
          world_T_sensor.inverse(),
          Eigen::Map<const Eigen::VectorXf>(vec.data(), vec.size()),
          GlobalInfo::instance().getSensor(packet.sensor_name).get(),
          depth_converted);
    }
    else
    {
      output_packet = std::make_unique<FeatureView>(
          timestamp_ns,
          world_T_sensor.inverse(),
          Eigen::Map<const Eigen::VectorXf>(vec.data(), vec.size()),
          GlobalInfo::instance().getSensor(packet.sensor_name).get());
    }

    // MODIFIED: Push to the new synchronized queue instead of the old features queue
    PipelineQueues::instance(scene_graph_id_).input_features_queue.push(std::move(output_packet));
  }
}


FeatureReceiver::FeatureReceiver(const Config& config, int scene_graph_id)
    : config(config::checkValid(config)), lookup_(config.tf_lookup), nh_(config.ns),
      raw_message_queue_(std::make_shared<MessageQueue<SyncedMessage>>()), scene_graph_id_(scene_graph_id) {}

FeatureReceiver::~FeatureReceiver() {}

void FeatureReceiver::stop() {
  should_shutdown_ = true;
  if (processing_thread_.joinable()) {
    processing_thread_.join();
  }
}

void FeatureReceiver::save(const LogSetup&) {}

std::string FeatureReceiver::printInfo() const { return config::toString(config); }

}  // namespace hydra
