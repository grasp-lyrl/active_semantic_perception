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
#include "hydra_ros/hydra_ros_pipeline.h"

#include <config_utilities/config.h>
#include <config_utilities/parsing/ros.h>
#include <config_utilities/printing.h>
#include <config_utilities/validation.h>
#include <hydra/active_window/reconstruction_module.h>
#include <hydra/backend/backend_module.h>
#include <hydra/backend/zmq_interfaces.h>
#include <hydra/common/dsg_types.h>
#include <hydra/common/global_info.h>
#include <hydra/frontend/graph_builder.h>
#include <hydra/loop_closure/loop_closure_module.h>
#include <pose_graph_tools_ros/conversions.h>

#include <memory>

#include "hydra_ros/backend/ros_backend_publisher.h"
#include "hydra_ros/frontend/ros_frontend_publisher.h"
#include "hydra_ros/loop_closure/ros_lcd_registration.h"
#include "hydra_ros/utils/bow_subscriber.h"

namespace hydra {

void declare_config(HydraRosPipeline::Config& config) {
  using namespace config;
  name("HydraRosConfig");
  field(config.active_window, "active_window");
  field(config.frontend, "frontend");
  field(config.backend, "backend");
  field(config.enable_frontend_output, "enable_frontend_output");
  field(config.input, "input");
  config.features.setOptional();
  field(config.features, "features");
}

HydraRosPipeline::HydraRosPipeline(const ros::NodeHandle& nh, int robot_id, int num_scene_graph)
    : HydraPipeline(config::fromRos<PipelineConfig>(nh), robot_id, num_scene_graph),
      config(config::checkValid(config::fromRos<Config>(nh))),
      nh_(nh) {
  LOG(INFO) << "Starting Hydra-ROS with input configuration\n"
            << config::toString(config.input);
}

HydraRosPipeline::~HydraRosPipeline() {}

void HydraRosPipeline::init() {
  // Overall idea: create only one input but multiple frontend and backend modules
  const auto& pipeline_config = GlobalInfo::instance().getConfig();
  const auto logs = GlobalInfo::instance().getLogs();
  std::vector<std::shared_ptr<MessageQueue<InputPacket::Ptr>>> active_window_queues;
  // Ensemble of scene graphs
  LOG(INFO) << "Creating " << num_scene_graph_ << " scene graphs";
  for (int i = 0; i < num_scene_graph_; ++i) {
    frontend_.emplace_back(config.frontend.create(std::get<0>(scene_graph_info_[i]), std::get<2>(scene_graph_info_[i]), logs, i));
    modules_["frontend" + std::to_string(i)] = CHECK_NOTNULL(frontend_[i]);
    backend_.emplace_back(config.backend.create(std::get<1>(scene_graph_info_[i]), std::get<2>(scene_graph_info_[i]), logs, i));
    modules_["backend" + std::to_string(i)] = CHECK_NOTNULL(backend_[i]);
    active_window_.emplace_back(config.active_window.create(frontend_[i]->queue(), i));
    active_window_queues.push_back(active_window_[i]->queue());
    modules_["active_window" + std::to_string(i)] = CHECK_NOTNULL(active_window_[i]);
    if (pipeline_config.enable_lcd) {
      initLCD(i);
      bow_sub_[i].reset(new BowSubscriber(nh_, i));
    }
    ros::NodeHandle bnh(nh_, "graph" + std::to_string(i) + "/backend");
    backend_[i]->addSink(std::make_shared<RosBackendPublisher>(bnh));
    auto zmq_config = config::fromRos<ZmqSink::Config>(bnh, "zmq_sink");

    // Modify the address to make it compatible with multiple scene graphs
    size_t colon_pos = zmq_config.url.find_last_of(':');
    std::string port_str = zmq_config.url.substr(colon_pos + 1);
    int port_num = std::stoi(port_str);
    port_num += 2 * i;
    std::string new_url = zmq_config.url.substr(0, colon_pos + 1) + std::to_string(port_num);
    zmq_config.url = new_url;

    backend_[i]->addSink(std::make_shared<ZmqSink>(zmq_config));
    if (config.enable_frontend_output) {
      CHECK(frontend_[i]) << "Frontend module required!";
      frontend_[i]->addSink(
          std::make_shared<RosFrontendPublisher>(ros::NodeHandle(nh_, "graph" + std::to_string(i) + "/frontend")));
    }
  }
  input_module_ =
      std::make_shared<RosInputModule>(config.input, active_window_queues);
  if (config.features) {
    for (int i = 0; i < num_scene_graph_; ++i) {
      modules_["features" + std::to_string(i)] = config.features.create(i);
    }
  }

  // modules_["backend"] = CHECK_NOTNULL(backend_);

  // frontend_ = config.frontend.create(frontend_dsg_, shared_state_, logs);
  // modules_["frontend"] = CHECK_NOTNULL(frontend_);

  // active_window_ = config.active_window.create(frontend_->queue());
  // modules_["active_window"] = CHECK_NOTNULL(active_window_);

  // if (pipeline_config.enable_lcd) {
  //   initLCD();
  //   bow_sub_.reset(new BowSubscriber(nh_));
  // }

  // ros::NodeHandle bnh(nh_, "backend");
  // backend_->addSink(std::make_shared<RosBackendPublisher>(bnh));
  // const auto zmq_config = config::fromRos<ZmqSink::Config>(bnh, "zmq_sink");
  // backend_->addSink(std::make_shared<ZmqSink>(zmq_config));

  // if (config.enable_frontend_output) {
  //   CHECK(frontend_) << "Frontend module required!";
  //   frontend_->addSink(
  //       std::make_shared<RosFrontendPublisher>(ros::NodeHandle(nh_, "frontend")));
  // }

  // input_module_ =
  //     std::make_shared<RosInputModule>(config.input, active_window_->queue());
  // if (config.features) {
  //   modules_["features"] = config.features.create();  // has to come after input module
  // }
}

void HydraRosPipeline::stop() {
  // enforce stop order to make sure every data packet is processed
  input_module_->stop();
  // TODO(nathan) push extracting active window objects to module stop
  for (const auto& active_window : active_window_) {
    active_window->stop();
  }
  for (const auto& frontend : frontend_) {
    frontend->stop();
  }
  for (const auto& backend : backend_) {
    backend->stop();
  }

  HydraPipeline::stop();
}

void HydraRosPipeline::initLCD(int scene_graph_id) {
  auto lcd_config = config::fromRos<LoopClosureConfig>(nh_);
  lcd_config.detector.num_semantic_classes = GlobalInfo::instance().getTotalLabels();
  VLOG(1) << "Number of classes for LCD: " << lcd_config.detector.num_semantic_classes;
  config::checkValid(lcd_config);

  auto lcd = std::make_shared<LoopClosureModule>(lcd_config, shared_state_, scene_graph_id);
  modules_["lcd" + std::to_string(scene_graph_id)] = lcd;

  if (lcd_config.detector.enable_agent_registration) {
    lcd->getDetector().setRegistrationSolver(0,
                                             std::make_unique<lcd::DsgAgentSolver>());
  }
}

}  // namespace hydra
