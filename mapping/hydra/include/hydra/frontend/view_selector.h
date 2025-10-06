#pragma once
#include <spark_dsg/node_attributes.h>

#include "hydra/openset/openset_types.h"
#include <opencv2/opencv.hpp>

namespace hydra {

class Sensor;

struct FeatureView {
  using Ptr = std::unique_ptr<FeatureView>;
  FeatureView(uint64_t timestamp_ns,
              const Eigen::Isometry3d& sensor_T_world,
              const FeatureVector& feature,
              const Sensor* sensor,
              const cv::Mat& depth_image);
    
  FeatureView(uint64_t timestamp_ns,
              const Eigen::Isometry3d& sensor_T_world,
              const FeatureVector& feature,
              const Sensor* sensor);

  const uint64_t timestamp_ns;
  const Eigen::Isometry3d sensor_T_world;
  const FeatureVector feature;
  cv::Mat depth_image;

  const Sensor& sensor() const;
  bool pointInView(const Eigen::Vector3d& point_w,
                    const cv::Mat& depth_image,
                    Eigen::Vector3d* point_s = nullptr) const;

 private:
  const Sensor* const sensor_;
};

struct ViewSelector {
  using FeatureList = std::list<FeatureView::Ptr>;
  virtual ~ViewSelector() = default;
  virtual void selectFeature(const FeatureList& views,
                             spark_dsg::SemanticNodeAttributes& attrs) const = 0;
};

}  // namespace hydra
