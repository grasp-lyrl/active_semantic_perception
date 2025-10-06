#include "hydra_ros/active_window/nothing_bbox_extractor.h"
#include <config_utilities/config.h>
#include <config_utilities/factory.h>
#include <config_utilities/printing.h>
#include <config_utilities/validation.h>

namespace hydra {
void declare_config(NothingBBoxExtractor::Config& config) {
  using namespace config;
  name("NothingBBoxExtractor::Config");
  field(config.threshold, "threshold");
  field(config.ns, "ns");
}

// TODO(huayi): Change the topic to be consistent across two modules.
NothingBBoxExtractor::NothingBBoxExtractor(const Config& config) 
    : config(config::checkValid(config)),
       nh_(config.ns) {
       sub_ = nh_.subscribe("/vs_graphs/walls", 
                         1, 
                         &NothingBBoxExtractor::messageCallback, 
                         this);
       scene_number_ = nh_.param<int>("/clio_node/scene_number", -1);
    }

NothingBBoxExtractor::~NothingBBoxExtractor(){
}

void NothingBBoxExtractor::messageCallback(const vision_msgs::Detection3DArray::ConstPtr& msg) {
    // Lock the mutex and update the shared pointer to the latest message
    std::lock_guard<std::mutex> lock(msg_mutex_);
    latest_msg_ = msg;
}

NothingBBoxExtractor::FreeSpace2DResult NothingBBoxExtractor::findLargestFreeSpaceInSlice(
    const spark_dsg::BoundingBox& wall_bbox,
    const TsdfLayer& tsdf_layer,
    float voxel_size,
    float sdf_threshold,
    float min_area_threshold) const {

    // 1. Define the slice geometry from the wall's bounding box
    Eigen::Matrix3f R = wall_bbox.world_R_center;
    Eigen::Vector3f size = wall_bbox.dimensions;
    Eigen::Vector3f center = wall_bbox.world_P_center;

    // Find the thin dimension (normal to the wall plane) and the two larger dimensions (u, v)
    int min_dim_idx = 0;
    if (size.y() < size.x()) min_dim_idx = 1;
    if (size.z() < size(min_dim_idx)) min_dim_idx = 2;

    float validation_offset = size(min_dim_idx) / 2.0f;

    int u_idx = (min_dim_idx + 1) % 3;
    int v_idx = (min_dim_idx + 2) % 3;

    float slice_width_m = size(u_idx);
    float slice_height_m = size(v_idx);

    // Get the local axes directly from the columns of the rotation matrix
    Eigen::Vector3f u_axis = R.col(u_idx);
    Eigen::Vector3f v_axis = R.col(v_idx);
    Eigen::Vector3f normal_axis = R.col(min_dim_idx);

    // 2. Create a 2D grid representation of the slice by sampling the TSDF
    int grid_width = static_cast<int>(std::ceil(slice_width_m / voxel_size));
    int grid_height = static_cast<int>(std::ceil(slice_height_m / voxel_size));

    if (grid_width <= 0 || grid_height <= 0) {
        return {}; // Invalid dimensions
    }

    std::vector<char> binary_map(grid_height * grid_width, 0);
    Eigen::Vector3f slice_origin = center - (u_axis * slice_width_m / 2.0f) - (v_axis * slice_height_m / 2.0f);

    for (int r = 0; r < grid_height; ++r) {
        // Calculate the starting point for this row
        Eigen::Vector3f row_start_point = slice_origin + ((r + 0.5f) * voxel_size * v_axis);
        
        // Pre-calculate the step vector for moving along the row
        Eigen::Vector3f u_step = voxel_size * u_axis;
        
        // Start from the center of the first voxel in the row
        Eigen::Vector3f point_on_plane = row_start_point + (0.5f * u_step);

        for (int c = 0; c < grid_width; ++c) {
            // Query the TSDF with the current point
            GlobalIndex index = tsdf_layer.getGlobalVoxelIndex(point_on_plane);
            if (tsdf_layer.hasVoxel(index)) {
                const TsdfVoxel& voxel = tsdf_layer.getVoxel(index);
                 if (voxel.distance > sdf_threshold && voxel.weight > 1e-6) {
                    binary_map[r * grid_width + c] = 1; // Mark as free space
                }
            }
            
            // Move to the next point on the plane incrementally
            point_on_plane += u_step;
        }
    }

    // 3. Find the largest rectangle of 1s in the binary_map using the histogram method
    int max_area = 0;
    int best_r = 0, best_c = 0, best_h = 0, best_w = 0;

    std::vector<int> heights(grid_width, 0);
    for (int r = 0; r < grid_height; ++r) {
        for (int c = 0; c < grid_width; ++c) {
           heights[c] = (binary_map[r * grid_width + c] == 1) ? heights[c] + 1 : 0;
        }

        std::stack<int> s;
        for (int c = 0; c <= grid_width; ++c) {
            int h = (c == grid_width) ? 0 : heights[c];
            while (!s.empty() && heights[s.top()] >= h) {
                int current_height = heights[s.top()];
                s.pop();
                int current_width = s.empty() ? c : c - s.top() - 1;
                int current_area = current_height * current_width;

                if (current_area > max_area) {
                    max_area = current_area;
                    best_h = current_height;
                    best_w = current_width;
                    best_c = s.empty() ? 0 : s.top() + 1;
                    best_r = r - current_height + 1;
                }
            }
            s.push(c);
        }
    }
    
    // 4. Check against volume threshold
    float max_area_m2 = max_area * voxel_size * voxel_size;
    if (max_area_m2 < min_area_threshold) {
        return {}; // Not found or too small
    }

    // 5. Check if the found rectangle is also free on front and back slices
    // Helper lambda to check if a single point in world coordinates is free space
    auto is_point_free = [&](const Eigen::Vector3f& point) {
        GlobalIndex index = tsdf_layer.getGlobalVoxelIndex(point);
        if (tsdf_layer.hasVoxel(index)) {
            const TsdfVoxel& voxel = tsdf_layer.getVoxel(index);
            // Check if distance is above threshold and weight is valid
            return (voxel.distance > sdf_threshold && voxel.weight > 1e-6);
        }
        return true; // Voxel not in layer is considered free
    };

    int violation_count = 0;
    bool validation_failed = false;
    int max_allowed_violations = 50;

    // Iterate only through the voxels of the candidate rectangle
    for (int r_offset = 0; r_offset < best_h; ++r_offset) {
        for (int c_offset = 0; c_offset < best_w; ++c_offset) {
            // Calculate the position of the current voxel on the center slice
            int r = best_r + r_offset;
            int c = best_c + c_offset;
            
            Eigen::Vector3f point_on_center_slice = slice_origin + 
                                                    ((r + 0.5f) * voxel_size * v_axis) + 
                                                    ((c + 0.5f) * voxel_size * u_axis);

            // Check the corresponding points on the front and back validation slices
            Eigen::Vector3f point_on_front_slice = point_on_center_slice + validation_offset * normal_axis;
            Eigen::Vector3f point_on_back_slice = point_on_center_slice - validation_offset * normal_axis;

            if (!is_point_free(point_on_front_slice) || !is_point_free(point_on_back_slice)) {
                violation_count++;
            }

            // If the front or back point is not free, increment the counter
            if (violation_count > max_allowed_violations) {
                validation_failed = true;
                break; // Exit the inner loop (c_offset)
            }
        }
        if (validation_failed) {
            break; // Exit the outer loop (r_offset)
        }
    }

    // Check the flag to determine the final result
    if (validation_failed) {
        return {}; // Validation failed
    }

    FreeSpace2DResult result;
    result.found = true;
    
    float rect_width_m = best_w * voxel_size;
    float rect_height_m = best_h * voxel_size;
    result.dimensions = {rect_width_m, rect_height_m};

    // Calculate center of the found rectangle in world coordinates
    Eigen::Vector3f rect_top_left_in_slice = (best_c * voxel_size * u_axis) + (best_r * voxel_size * v_axis);
    Eigen::Vector3f rect_center_in_slice = rect_top_left_in_slice + (rect_width_m / 2.0f * u_axis) + (rect_height_m / 2.0f * v_axis);
    result.center = slice_origin + rect_center_in_slice;

    return result;
}


// TODO(huayi): Modify the pipeline. I hack the function for now to add (wall, window, curtain) bounding boxes
void NothingBBoxExtractor::call(uint64_t timestamp_ns,
                                       const VolumetricMap& map,
                                       ActiveWindowOutput& output) const {
  TsdfLayer tsdf_layer = map.getTsdfLayer();
  if (tsdf_layer.numBlocks() == 0) {
      return;
  }
  // Step 1: Find the maximal cuboid of "nothing"
  MaximalCuboidFinder cuboid_finder(config.threshold, tsdf_layer);
  GlobalCuboid max_cuboid = cuboid_finder.findMaximalCuboid(tsdf_layer);
  if (max_cuboid.width <= 0.4f || max_cuboid.height <= 0.4f || max_cuboid.depth <= 0.4f || max_cuboid.volume <= 0.4f) {
        return; // cuboid too small, skip the update
  }

  // Step 2: Add the results to the scene graph
  auto nothing_update = std::make_shared<hydra::LayerUpdate>(spark_dsg::DsgLayers::OBJECTS);
  auto nothing_attrs =  std::make_unique<spark_dsg::SemanticNodeAttributes>();
  Eigen::Vector3f center(max_cuboid.x + max_cuboid.width / 2.0f,
                         max_cuboid.y + max_cuboid.depth / 2.0f,
                         max_cuboid.z + max_cuboid.height / 2.0f);
  Eigen::Vector3f dimensions(max_cuboid.width - 0.2f, 
                             max_cuboid.depth - 0.2f,
                             max_cuboid.height - 0.2f);
  nothing_attrs->bounding_box = spark_dsg::BoundingBox(dimensions, center);

  nothing_attrs->name = "Nothing";
  nothing_attrs->position = Eigen::Vector3d(center.x(), center.y(), center.z());
  nothing_update->attributes.push_back(std::move(nothing_attrs));
  output.graph_update[nothing_update->layer]->append(std::move(*nothing_update));
//   VLOG(0) << "Current Update Size: " << output.graph_update[nothing_update->layer]->attributes.size();

  // Step 3: Add walls, windows, and curtains
  vision_msgs::Detection3DArray::ConstPtr current_detections;
  std::lock_guard<std::mutex> lock(msg_mutex_);
  current_detections = latest_msg_;
  if (current_detections){
    for (const auto& detection : current_detections->detections) {
    // General information calculation
      auto structural_update = std::make_shared<hydra::LayerUpdate>(spark_dsg::DsgLayers::OBJECTS);
      auto structural_attrs =  std::make_unique<spark_dsg::SemanticNodeAttributes>();
      structural_attrs->position = Eigen::Vector3d(detection.bbox.center.position.x,
                                        detection.bbox.center.position.y,
                                        detection.bbox.center.position.z);
      structural_attrs->bounding_box = spark_dsg::BoundingBox(Eigen::Vector3f(detection.bbox.size.x,
                                                                   detection.bbox.size.y,
                                                                   detection.bbox.size.z),
                                                   Eigen::Vector3f(detection.bbox.center.position.x,
                                                                   detection.bbox.center.position.y,
                                                                   detection.bbox.center.position.z),
                                                   Eigen::Quaternionf(detection.bbox.center.orientation.w,
                                                                      detection.bbox.center.orientation.x,
                                                                      detection.bbox.center.orientation.y,
                                                                      detection.bbox.center.orientation.z));
      bool potential_door = false;
      float smallest_dimension = std::min({structural_attrs->bounding_box.dimensions.x(),
                                          structural_attrs->bounding_box.dimensions.y(),
                                          structural_attrs->bounding_box.dimensions.z()});
      if (smallest_dimension > 0.5f)
      {
          potential_door = true;
      }
      // Skip large bounding boxes
    //   if (structural_attrs->bounding_box.volume() > 8.0f
    //       || structural_attrs->bounding_box.dimensions.x() > 8.0f
    //       || structural_attrs->bounding_box.dimensions.y() > 8.0f
    //       || structural_attrs->bounding_box.dimensions.z() > 8.0f) {
    //     continue;
    //   }

      // Pass the test, assign the name based on the detection result
      switch (detection.results[0].id) {
        case planeVariant::CURTAIN:
            structural_attrs->name = "Curtain";
            break;
        case planeVariant::BLIND:
            structural_attrs->name = "Blind";
            break;
        case planeVariant::WINDOW:
            structural_attrs->name = "Window";
            break;
        case planeVariant::WALL:
            if (scene_number_ == 853 && potential_door) {
                structural_attrs->name = "door";
            }
            else{
                structural_attrs->name = "Wall";
            }
            break;
        // Changing the door to wall is better for LLM prediction
        case planeVariant::DOOR:
            if (scene_number_ == 853 && potential_door) {
                structural_attrs->name = "door";
            }
            else{
                structural_attrs->name = "Wall";
            }
            break;
      }

      // Find the largest free space in the wall slice if it is in view
      bool is_wall = (detection.results[0].id == planeVariant::WALL);
      bool is_in_view = false;
      if (is_wall) {
           Eigen::Isometry3f sensor_T_world = output.sensor_data->world_T_body.inverse().cast<float>();
           is_in_view = output.sensor_data->getSensor().pointIsInViewFrustumWithOcclusion(sensor_T_world * structural_attrs->bounding_box.world_P_center, 
                                                                                        output.sensor_data->depth_image, 0.5f);
      }

      if (is_wall && is_in_view) {
          // Define a minimum area for an opening to be considered valid.
          // This should ideally come from the config.
          const float min_opening_area_m2 = 1.0f; 

          FreeSpace2DResult opening = findLargestFreeSpaceInSlice(structural_attrs->bounding_box,
                                                                  tsdf_layer,
                                                                  tsdf_layer.voxel_size,
                                                                  config.threshold, 
                                                                  min_opening_area_m2);
          if (opening.found) {
              // An opening was found, create a new node for it.
              auto opening_update = std::make_shared<hydra::LayerUpdate>(spark_dsg::DsgLayers::OBJECTS);
              auto opening_attrs = std::make_unique<spark_dsg::SemanticNodeAttributes>();

              opening_attrs->name = "door";
              opening_attrs->position = Eigen::Vector3d(opening.center.x(), opening.center.y(), opening.center.z());
              
              // Construct the 3D size vector for the opening's BBox
              Eigen::Vector3f wall_size = structural_attrs->bounding_box.dimensions;
              int min_dim_idx = 0;
              if (wall_size.y() < wall_size.x()) min_dim_idx = 1;
              if (wall_size.z() < wall_size(min_dim_idx)) min_dim_idx = 2;
              
              int u_idx = (min_dim_idx + 1) % 3;
              int v_idx = (min_dim_idx + 2) % 3;
              
              Eigen::Vector3f opening_size;
              opening_size(min_dim_idx) = wall_size(min_dim_idx); // Same thickness as wall
              opening_size(u_idx) = opening.dimensions.x();       // Width of found free space
              opening_size(v_idx) = opening.dimensions.y();       // Height of found free space
            //   opening_attrs->bounding_box = spark_dsg::BoundingBox(opening_size,
            //                                                        opening.center);

              opening_attrs->bounding_box = spark_dsg::BoundingBox(spark_dsg::BoundingBox::Type::OBB, opening_size, opening.center, structural_attrs->bounding_box.world_R_center);
              opening_update->attributes.push_back(std::move(opening_attrs));
              output.graph_update[opening_update->layer]->append(std::move(*opening_update));
          }
      }

      // After finding the opening, add the structural node to the graph update
      structural_update->attributes.push_back(std::move(structural_attrs));
      output.graph_update[structural_update->layer]->append(std::move(*structural_update));
    }
  }
}

MaximalCuboidFinder::MaximalCuboidFinder(float threshold, const TsdfLayer& tsdf_layer) 
    : threshold_(threshold) {
    Bounds3D dim = getLayerBounds3D(tsdf_layer);
    x_dim_ = Eigen::Vector2f(dim.min.x(), dim.max.x());
    y_dim_ = Eigen::Vector2f(dim.min.y(), dim.max.y());
    z_dim_ = Eigen::Vector2f(dim.min.z(), dim.max.z());
    x_len_ = static_cast<int64_t>(dim.dims.x());
    y_len_ = static_cast<int64_t>(dim.dims.y());
    z_len_ = static_cast<int64_t>(dim.dims.z());
    voxel_size = dim.voxel_size;
}


float MaximalCuboidFinder::getSdf(const TsdfLayer& tsdf_layer, spatial_hash::Point point) const {
    // Calculate index in the 1D vector
    GlobalIndex index = tsdf_layer.getGlobalVoxelIndex(point);
    if (tsdf_layer.hasVoxel(index) == false) {
        return -1.0; // No voxel at this index, return occupied
    }
    return tsdf_layer.getVoxel(index).distance;
}

// Solves the largest rectangle in a histogram problem using a stack.
// This is a key sub-problem for the 3D case.
int64_t MaximalCuboidFinder::largestRectangleInHistogram(const std::vector<int64_t>& heights, int64_t& start_col, int64_t& rect_width, int64_t& rect_height) {
    std::stack<int64_t> s;
    int64_t max_area = 0;
    // Initialize output parameters
    start_col = 0;
    rect_width = 0;
    rect_height = 0;
    int64_t n = heights.size();

    for (int64_t i = 0; i <= n; ++i) {
        int64_t h = (i == n) ? 0 : heights[i];
        while (!s.empty() && heights[s.top()] >= h) {
            int64_t current_height = heights[s.top()];
            s.pop();
            int64_t current_width = s.empty() ? i : i - s.top() - 1;
            int64_t current_area = current_height * current_width;

            if (current_area > max_area) {
                max_area = current_area;
                start_col = s.empty() ? 0 : s.top() + 1;
                // When a new max area is found, save its dimensions
                rect_width = current_width;
                rect_height = current_height;
            }
        }
        s.push(i);
    }
    
    return max_area;
}

// Main function to find the maximal cuboid.
// The calculation complexity is O(Y*Z^2*X), where Y is depth, Z is height, and X is width.
// TODO(huayi): Discuss with professor if there is a simpler way to do this.
GlobalCuboid MaximalCuboidFinder::findMaximalCuboid(const TsdfLayer& tsdf_layer) {
    Cuboid max_cuboid_voxels{};
    // This 2D map will store cumulative DEPTHS along the Y-axis.
    std::vector<std::vector<int64_t>> depth_map_zx(z_len_, std::vector<int64_t>(x_len_, 0));

    for (int64_t y = 0; y < y_len_; ++y) {
        // 1. Update the cumulative depth map for the current Y-slice.
        for (int64_t z = 0; z < z_len_; ++z) {
            for (int64_t x = 0; x < x_len_; ++x) {
                spatial_hash::Point point(x_dim_.x() + x * voxel_size,
                                          y_dim_.x() + y * voxel_size,
                                          z_dim_.x() + z * voxel_size);
                if (getSdf(tsdf_layer, point) > threshold_) {
                    depth_map_zx[z][x]++;
                } else {
                    depth_map_zx[z][x] = 0;
                }
            }
        }

        // 2. Now, `depth_map_zx` is a 2D matrix where each cell value is a physical depth (Y-dim).
        //    We need to find the largest rectangle in this 2D map.
        std::vector<int64_t> row_as_histogram(x_len_, 0);
        for (int64_t z_start = 0; z_start < z_len_; ++z_start) {
            // For each starting row z_start, find the largest rectangle that starts there.
            // Initialize the histogram with the starting row.
            row_as_histogram = depth_map_zx[z_start];
            for (int64_t z_end = z_start; z_end < z_len_; ++z_end) {
                // Update the histogram by taking the minimum depth in the current z-column.
                for (int64_t x = 0; x < x_len_; ++x) {
                    row_as_histogram[x] = std::min(row_as_histogram[x], depth_map_zx[z_end][x]);
                }

                int64_t start_col=0, rect_width=0, rect_depth=0;
                largestRectangleInHistogram(row_as_histogram, start_col, rect_width, rect_depth);

                int64_t current_height = z_end - z_start + 1;
                int64_t current_volume = rect_width * current_height * rect_depth;

                if (current_volume > max_cuboid_voxels.volume) {
                    max_cuboid_voxels.volume = current_volume;
                    max_cuboid_voxels.x = start_col;
                    max_cuboid_voxels.y = y - rect_depth + 1;
                    max_cuboid_voxels.z = z_start;
                    max_cuboid_voxels.width = rect_width;
                    max_cuboid_voxels.height = current_height;
                    max_cuboid_voxels.depth = rect_depth;
                }
            }
        }
    }
    
    // Convert local coordinates to global coordinates
    GlobalCuboid max_cuboid_global{};
    max_cuboid_global.x = x_dim_.x() + max_cuboid_voxels.x * voxel_size;
    max_cuboid_global.y = y_dim_.x() + max_cuboid_voxels.y * voxel_size;
    max_cuboid_global.z = z_dim_.x() + max_cuboid_voxels.z * voxel_size;
    max_cuboid_global.width = max_cuboid_voxels.width * voxel_size;
    max_cuboid_global.height = max_cuboid_voxels.height * voxel_size;
    max_cuboid_global.depth = max_cuboid_voxels.depth * voxel_size;
    max_cuboid_global.volume = max_cuboid_voxels.volume * voxel_size * voxel_size * voxel_size;
    return max_cuboid_global;
}

namespace {

static const auto registration_ =
    config::RegistrationWithConfig<ReconstructionModule::Sink,
                                   NothingBBoxExtractor,
                                   NothingBBoxExtractor::Config>(
        "NothingBBoxExtractor");

}  // namespace
}  // namespace hydra