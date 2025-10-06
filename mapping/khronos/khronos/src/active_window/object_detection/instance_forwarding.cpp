/** -----------------------------------------------------------------------------
 * Copyright (c) 2024 Massachusetts Institute of Technology.
 * All Rights Reserved.
 *
 * AUTHORS:      Lukas Schmid <lschmid@mit.edu>, Marcus Abate <mabate@mit.edu>,
 *               Yun Chang <yunchang@mit.edu>, Luca Carlone <lcarlone@mit.edu>
 * AFFILIATION:  MIT SPARK Lab, Massachusetts Institute of Technology
 * YEAR:         2024
 * SOURCE:       https://github.com/MIT-SPARK/Khronos
 * LICENSE:      BSD 3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * -------------------------------------------------------------------------- */

#include "khronos/active_window/object_detection/instance_forwarding.h"

#include <string>
#include <vector>

#include "khronos/utils/geometry_utils.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>

namespace khronos {

void declare_config(InstanceForwarding::Config& config) {
  using namespace config;
  name("InstanceForwarding");
  field(config.verbosity, "verbosity");
  field(config.max_range, "max_range", "m");
  field(config.min_cluster_size, "min_cluster_size");
  field(config.max_cluster_size, "max_cluster_size");
  field(config.min_object_volume, "min_object_volume", "m");
  field(config.max_object_volume, "max_object_volume", "m");
}

InstanceForwarding::InstanceForwarding(const Config& config)
    : config(config::checkValid(config)),
      filter_by_volume_(config.min_object_volume > 0.0 || config.max_object_volume > 0.0) {}

void InstanceForwarding::processInput(const VolumetricMap& /* map */, FrameData& data) {
  processing_stamp_ = data.input.timestamp_ns;
  Timer timer("object_detection/all", processing_stamp_);

  extractSemanticClusters(data);
}

void InstanceForwarding::extractSemanticClusters(FrameData& data) {
  // Forward the semantic image from the input.
  // NOTE(lschmid): This assumes both images have the same type.
  data.object_image = data.input.label_image;

  // Extract clusters.
  std::unordered_map<int32_t, Pixels> clusters;
  for (int u = 0; u < data.input.label_image.cols; u++) {
    for (int v = 0; v < data.input.label_image.rows; v++) {
      const auto& id = data.input.label_image.at<InputData::LabelType>(v, u);
      // VLOG(0) << "Processing pixel at (" << u << ", " << v << ") with ID " << id;
      if (id == 0) {
        continue;
      }

      if (config.max_range > 0.f) {
        const float range = data.input.range_image.at<InputData::RangeType>(v, u);
        if (range > config.max_range) {
          continue;
        }
      }
      data.object_image.at<FrameData::ObjectImageType>(v, u) = id;
      clusters[id].emplace_back(u, v);
    }
  }

  for (const auto& [id, pixels] : clusters) {
    const auto curr_num_pixels = static_cast<int>(pixels.size());
    // VLOG(0) << "Processing cluster with ID " << id << " and " << curr_num_pixels
    //        << " pixels.";
    if (curr_num_pixels < config.min_cluster_size ||
        (config.max_cluster_size > 0 && curr_num_pixels > config.max_cluster_size)) {
      continue;
    }

    MeasurementCluster cluster;
    cluster.pixels.insert(cluster.pixels.end(), pixels.begin(), pixels.end());
    cluster.id = id;

    if (filter_by_volume_) {
      const auto bbox = BoundingBox(utils::VertexMapAdaptor(cluster.pixels, data.input.vertex_map));
      const auto volume = bbox.volume();
      if (volume < config.min_object_volume ||
          (config.max_object_volume > 0.0 && volume > config.max_object_volume)) {
        continue;
      }
    }

    // TODO(Yun) For now all semantic id is the same (so all label checks are invalid)
    const auto label = data.input.features[id - 1].class_name;
    // VLOG(0) << "Processing cluster with ID " << id << " and label '" << label
    //        << "' with " << curr_num_pixels << " pixels.";
    cluster.semantics = SemanticClusterInfo(label);
    // const auto feature = data.input.detect.find(id);
    // if (feature != data.input.label_features.end()) {
    //   cluster.semantics = SemanticClusterInfo(feature->second);
    // }

    data.semantic_clusters.emplace_back(std::move(cluster));
  }
}

// void InstanceForwarding::processBackground(cv::Mat& label_image)
// {
//     cv::Size imageSize = label_image.size();
//     cv::Mat markers = cv::Mat::zeros(imageSize, CV_32S);
//     cv::Mat combinedMask = label_image.clone();
//     // cv::Mat combinedMask = cv::Mat::zeros(imageSize, CV_8U);
//     for (int i = 0; i < instanceMasks.size(); ++i) {
//         // Ensure mask is of type CV_8U for logical operations
//         cv::Mat ucharMask;
//         instanceMasks[i].convertTo(ucharMask, CV_8U);

//         // Assign a unique ID (i+1) to each object instance in the markers image
//         markers.setTo(i + 1, ucharMask);

//         // 2. Create a combined binary mask of all objects
//         cv::bitwise_or(combinedMask, ucharMask, combinedMask);
//     }

//     cv::Mat imgForWatershed;
//     cv::cvtColor(combinedMask, imgForWatershed, cv::COLOR_GRAY2BGR);

//     cv::watershed(imgForWatershed, markers);

//     cv::Mat nothingInstances = markers.clone();
//     nothingInstances.setTo(0, combinedMask);
//     nothingInstances.setTo(0, markers == -1);

//     return nothingInstances;
// }
}  // namespace khronos
