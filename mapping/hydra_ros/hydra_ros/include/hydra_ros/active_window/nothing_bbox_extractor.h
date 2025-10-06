#pragma once
#include <hydra/active_window/reconstruction_module.h>
#include <hydra_ros/utils/occupancy_publisher.h>
#include <ros/ros.h>
#include <stack>
#include <vision_msgs/Detection3DArray.h>

namespace hydra {

class NothingBBoxExtractor : public ReconstructionModule::Sink {
 public:
  struct Config{
    float threshold = 0.24f;  // Threshold for the SDF value to consider a voxel as "nothing" (free space)
    std::string ns = "~";
    } const config;

  explicit NothingBBoxExtractor(const Config& config);

  virtual ~NothingBBoxExtractor();
  void call(uint64_t timestamp_ns,
            const VolumetricMap& map,
            ActiveWindowOutput& output) const override;
  void messageCallback(const vision_msgs::Detection3DArray::ConstPtr& msg);

  struct FreeSpace2DResult {
      bool found = false;
      Eigen::Vector3f center;
      Eigen::Vector2f dimensions; // width, height of the rectangle in the wall's plane
  };

  FreeSpace2DResult findLargestFreeSpaceInSlice(
      const spark_dsg::BoundingBox& wall_bbox,
      const TsdfLayer& tsdf_layer,
      float voxel_size,
      float sdf_threshold,
      float min_area_threshold) const;
  
 private:
  enum planeVariant
  {
      UNDEFINED = -1,
      CURTAIN = 0,
      BLIND = 1,
      WINDOW = 2,
      WALL = 3,
      DOOR = 4,
  };
  ros::NodeHandle nh_;
  ros::Subscriber sub_;
  vision_msgs::Detection3DArray::ConstPtr latest_msg_;
  mutable std::mutex msg_mutex_; // Mutable for locking in a const method
  mutable uint16_t nothing_box_id_ = 0; // ID for the "Nothing" bounding box
  int scene_number_ = -1; // Scene number
};

struct Cuboid {
  int x, y, z;
  int width, height, depth;
  int64_t volume;
};
struct GlobalCuboid {
  float x, y, z;
  float width, height, depth;
  float volume;
};

class MaximalCuboidFinder {
 public:
  MaximalCuboidFinder(float threshold, const TsdfLayer& tsdf_layer);
  float getSdf(const TsdfLayer& tsdf_layer, spatial_hash::Point point) const;

  int64_t largestRectangleInHistogram(const std::vector<int64_t>& heights, int64_t& start_col, int64_t& rect_width, int64_t& rect_height);

  GlobalCuboid findMaximalCuboid(const TsdfLayer& tsdf_layer);

 private:
    Eigen::Vector2f x_dim_;
    Eigen::Vector2f y_dim_;
    Eigen::Vector2f z_dim_;
    int64_t x_len_;
    int64_t y_len_;
    int64_t z_len_;
    float threshold_;
    float voxel_size; // Size of each voxel in the TSDF layer
};

void declare_config(NothingBBoxExtractor::Config& config);

}  // namespace hydra
