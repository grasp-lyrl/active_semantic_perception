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
#include "hydra/common/graph_update.h"

#include <config_utilities/config.h>
#include <config_utilities/types/conversions.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <spark_dsg/dynamic_scene_graph.h>

#include <algorithm>

#include "hydra/common/config_utilities.h"

namespace YAML {

template <typename T>
struct convert<std::optional<T>> {
  static Node encode(const std::optional<T>& opt) {
    if (opt) {
      return YAML::convert<T>::encode(opt.value());
    }

    Node node("");
    node.SetTag("null");
    return node;
  }

  static bool decode(const Node& node, std::optional<T>& opt) {
    if (node) {
      if (node.Tag() == "null") {
        return true;
      }

      opt = node.as<T>();
    }

    return true;
  }
};

}  // namespace YAML

namespace hydra {

using namespace spark_dsg;

void declare_config(LayerTracker::Config& config) {
  using namespace config;
  name("LayerTracker::Config");
  field<CharConversion>(config.prefix, "prefix");
  field(config.target_layer, "target_layer");
}

void declare_config(GraphUpdater::Config& config) {
  using namespace config;
  name("GraphUpdater::Config");
  field(config.layer_updates, "layer_updates");
  field(config.duplicate_score, "duplicate_score");
  field(config.duplicate_score_structural, "duplicate_score_structural");
  field(config.nothing_decomposition_threshold, "nothing_decomposition_threshold");
}

LayerUpdate::LayerUpdate(spark_dsg::LayerId layer) : layer(layer) {}

void LayerUpdate::append(LayerUpdate&& rhs) {
  if (layer != rhs.layer) {
    return;
  }

  std::move(
      rhs.attributes.begin(), rhs.attributes.end(), std::back_inserter(attributes));
  rhs.attributes.clear();
}

LayerTracker::LayerTracker(const Config& config)
    : config(config), next_id(config.prefix, 0) {}

GraphUpdater::GraphUpdater(const Config& config) : config(config::checkValid(config)) {
  for (const auto& [layer_name, tracker_config] : config.layer_updates) {
    trackers_.emplace(DsgLayers::StringToLayerId(layer_name),
                      LayerTracker(tracker_config));
  }
}

// TODO(huayi): Not a decent solution to add walls, windows, curtains and blinds
void GraphUpdater::update(const GraphUpdate& update, DynamicSceneGraph& graph) {
  bool update_contains_walls = false;
  for (const auto& [layer_id, layer_update] : update) {
    if (layer_update) {
      for (const auto& attrs : layer_update->attributes) {
        // Safely cast to check the attribute's name.
        if (const auto* semantic_attrs = dynamic_cast<const SemanticNodeAttributes*>(attrs.get())) {
          if (semantic_attrs->name == "Wall") {
            update_contains_walls = true;
            break; // Found a wall, no need to check further.
          }
        }
      }
    }
    if (update_contains_walls) {
      break; // Exit the outer loop as well.
    }
  }

  // 1. If the update contains walls, delete all old tracked structural nodes.
  //    Otherwise, do nothing and keep the old nodes.
  if (update_contains_walls) {
    for (const auto& node_id : wall_id_trackers_) {
      if (graph.hasNode(node_id)) {
        graph.removeNode(node_id);
      }
    }
    wall_id_trackers_.clear();
  }

  // 2. Process the graph updates: Check overlap and decompose nothing object bounding boxes
  for (const auto& [layer_id, layer_update] : update) {
    if (!layer_update) {
      LOG(WARNING) << "Received invalid update for layer " << layer_id;
    }

    auto iter = trackers_.find(layer_id);
    if (iter == trackers_.end()) {
      LOG(WARNING) << "Received updates for unhandled layer " << layer_id;
      return;
    }

    // This vector will store all structural objects added in this update
    std::vector<BoundingBox> added_boxes_this_update;

    auto& tracker = iter->second;
    for (auto&& attrs : layer_update->attributes) {
      // VLOG(0) << "Emplacing " << tracker.next_id.getLabel() << " @ "
      //         << tracker.config.target_layer.value_or(layer_id) << " for layer "
      //         << layer_id;
      auto* semantic_attrs = dynamic_cast<SemanticNodeAttributes*>(attrs.get());
      std::vector<BoundingBox> decomposed_boxes = removeInactiveObjects(semantic_attrs, graph, tracker.config.target_layer.value_or(layer_id));

      for (const BoundingBox& box : decomposed_boxes) {
        // Create a new node for each decomposed box.
        auto new_attrs = std::make_unique<SemanticNodeAttributes>();
        new_attrs->bounding_box = box;
        new_attrs->name = "Nothing";
        new_attrs->position = Eigen::Vector3d(box.world_P_center.x(), box.world_P_center.y(), box.world_P_center.z());
        graph.emplaceNode(tracker.config.target_layer.value_or(layer_id), tracker.next_id, std::move(new_attrs));
        ++tracker.next_id;
      }

      // Save the id if the object is a structural object
      bool is_structural_object = (semantic_attrs->name == "Wall" || 
                                     semantic_attrs->name == "Window" || 
                                     semantic_attrs->name == "Curtain" || 
                                     semantic_attrs->name == "Blind");

      if (is_structural_object) {
        wall_id_trackers_.push_back(static_cast<NodeId>(tracker.next_id));
      }

      graph.emplaceNode(tracker.config.target_layer.value_or(layer_id),
                        tracker.next_id,
                        std::move(attrs));
      ++tracker.next_id;
    }

    // 3. Merge "Nothing" bounding boxes if they are adjacent
    std::vector<BoundingBox> node_to_add;
    std::vector<size_t> node_to_delete;
    auto& layer = graph.getLayer(tracker.config.target_layer.value_or(layer_id));
    for (auto iter1 = layer.nodes().cbegin(); iter1 != layer.nodes().cend(); ++iter1) {
      const auto& node1 = iter1->second;
      
      // Start the inner loop from the next node to avoid duplicate comparisons
      auto iter2 = iter1;
      ++iter2;
      for (; iter2 != layer.nodes().cend(); ++iter2) {
        const auto& node2 = iter2->second;
        
        const auto& attrs1 = node1->attributes<SemanticNodeAttributes>();
        const auto& attrs2 = node2->attributes<SemanticNodeAttributes>();

        // Check if two "nothing" bounding boxes can be merged
        if (attrs1.name == "Nothing" && attrs2.name == "Nothing") {
          auto merged_box = mergeBoxes(attrs1.bounding_box, attrs2.bounding_box);
          if (merged_box && 
              std::find(node_to_delete.begin(), node_to_delete.end(), node1->id) == node_to_delete.end() && 
              std::find(node_to_delete.begin(), node_to_delete.end(), node2->id) == node_to_delete.end()) {
            node_to_add.push_back(*merged_box);
            node_to_delete.push_back(node1->id);
            node_to_delete.push_back(node2->id);
          }
        }
        if (attrs1.name == "door" && (attrs2.name == "Window" || attrs2.name == "Curtain" || attrs2.name == "Blind") && (attrs1.bounding_box.world_P_center - attrs2.bounding_box.world_P_center).norm() < 1.0f) {
          // Delete door if it is close to window/curtain/blind
            node_to_delete.push_back(node1->id);
        }
        else if ((attrs1.name == "Window" || attrs1.name == "Curtain" || attrs1.name == "Blind") && attrs2.name == "door" && (attrs1.bounding_box.world_P_center - attrs2.bounding_box.world_P_center).norm() < 1.0f) {
            // Delete door if it is close to window/curtain/blind
            node_to_delete.push_back(node2->id);
        }
      }
    }

    for (size_t node_id : node_to_delete) {
      graph.removeNode(node_id);
    }
    for (const BoundingBox& box : node_to_add) {
      // Create a new node for each merged box.
      auto new_attrs = std::make_unique<SemanticNodeAttributes>();
      new_attrs->bounding_box = box;
      new_attrs->name = "Nothing";
      new_attrs->position = Eigen::Vector3d(box.world_P_center.x(), box.world_P_center.y(), box.world_P_center.z());
      graph.emplaceNode(tracker.config.target_layer.value_or(layer_id), tracker.next_id, std::move(new_attrs));
      ++tracker.next_id;
    }
  }
}

std::vector<spark_dsg::BoundingBox> GraphUpdater::removeInactiveObjects(SemanticNodeAttributes* attr, DynamicSceneGraph& graph, LayerId layer_id) {
  // This function removes inactive objects from the graph and decomposes nothing objects based on IoU scores.
  BoundingBox& new_bbox = attr->bounding_box;
  auto& layer = graph.getLayer(layer_id);
  std::vector<size_t> node_ids_to_remove;
  std::vector<BoundingBox> decomposed_boxes;
  for (auto iter = layer.nodes().cbegin(); iter != layer.nodes().cend(); ++iter) {
    const auto& node = iter->second;
    SemanticNodeAttributes& semantic_attrs = node->attributes<SemanticNodeAttributes>();
    BoundingBox& node_bbox = semantic_attrs.bounding_box;
    // This calculation of score is only suitable for AABB bounding boxes
    float score = new_bbox.computeIoU(node_bbox);
    if (score > config.duplicate_score && semantic_attrs.name != "Nothing" && attr->name != "Nothing")
    {
      node_ids_to_remove.push_back(node->id);
    }
    else if (score > config.nothing_decomposition_threshold && semantic_attrs.name == "Nothing" && attr->name == "Nothing") {
      node_ids_to_remove.push_back(node->id);
    }
    else if (score > 0.0f && semantic_attrs.name == "Nothing" && attr->name == "Nothing")
    {
      // If the new bounding box is smaller than the old one, we decompose it
      // into smaller boxes to avoid removing the entire "Nothing" object.
      std::vector<BoundingBox> decomposed_box = decomposeBoundingBox(node_bbox, new_bbox);
      node_ids_to_remove.push_back(node->id);
      decomposed_boxes.insert(decomposed_boxes.end(), std::make_move_iterator(decomposed_box.begin()), std::make_move_iterator(decomposed_box.end()));
    }

    // Door uses OBB bounding box, so we need to use a different IoU calculation
    if (semantic_attrs.name == "door" && attr->name == "door") {
      score = new_bbox.computeIoU_OBB_ZAligned(node_bbox);
      if (score > config.duplicate_score_structural) {
        node_ids_to_remove.push_back(node->id);
      }
    }

    bool same_type_structural_object = ((attr->name == "Wall" && semantic_attrs.name == "Wall") || 
                                     (attr->name == "Window" && semantic_attrs.name == "Window") || 
                                     (attr->name == "Curtain" && semantic_attrs.name == "Curtain") || 
                                     (attr->name == "Blind" && semantic_attrs.name == "Blind"));

    // Structural objects also uses OBB bounding box
    if (same_type_structural_object) {
      score = new_bbox.computeIoU_OBB_ZAligned(node_bbox);
      if (score > config.duplicate_score_structural) {
        node_ids_to_remove.push_back(node->id);
        // Remove this object also from wall_id_trackers_
        auto it = std::find(wall_id_trackers_.begin(), wall_id_trackers_.end(), node->id);
        if (it != wall_id_trackers_.end()) {
          wall_id_trackers_.erase(it);
        }
      }
    }
  }

  for (size_t node_id : node_ids_to_remove) {
    graph.removeNode(node_id);
  }
  return decomposed_boxes;
}

std::vector<BoundingBox> GraphUpdater::decomposeBoundingBox(const BoundingBox& bbox_decompose, const BoundingBox& bbox_keep) {
    std::vector<BoundingBox> resultBoxes;

    Eigen::Vector3f newMin = bbox_decompose.minCorner();
    Eigen::Vector3f newMax = bbox_decompose.maxCorner();
    Eigen::Vector3f oldMin = bbox_keep.minCorner();
    Eigen::Vector3f oldMax = bbox_keep.maxCorner();

    // --- A helper lambda to create sub-boxes from their min/max corners ---
    auto createBoxFromCorners = [&](const Eigen::Vector3f& minCorner, const Eigen::Vector3f& maxCorner) {
        Eigen::Vector3f dims = maxCorner - minCorner;
        Eigen::Vector3f center = minCorner + dims / 2.0f;
        BoundingBox box(dims, center);
        if (box.dimensions.x() > 0.4f && box.dimensions.y() > 0.4f && box.dimensions.z() > 0.4f && box.volume() > 0.4f) {
            resultBoxes.push_back(box);
        }
    };

    // Calculate the six potential resulting boxes.

    // Top box (positive Y direction)
    createBoxFromCorners({newMin.x(), oldMax.y(), newMin.z()}, newMax);
    
    // Bottom box (negative Y direction)
    createBoxFromCorners(newMin, {newMax.x(), oldMin.y(), newMax.z()});

    // Clipped Y-range for the remaining boxes
    float clipped_y1 = std::max(newMin.y(), oldMin.y());
    float clipped_y2 = std::min(newMax.y(), oldMax.y());

    // Left box (negative X direction)
    createBoxFromCorners({newMin.x(), clipped_y1, newMin.z()}, {oldMin.x(), clipped_y2, newMax.z()});
    
    // Right box (positive X direction)
    createBoxFromCorners({oldMax.x(), clipped_y1, newMin.z()}, {newMax.x(), clipped_y2, newMax.z()});

    // Clipped X and Y-range for front/back boxes
    float clipped_x1 = std::max(newMin.x(), oldMin.x());
    float clipped_x2 = std::min(newMax.x(), oldMax.x());

    // Front box (positive Z direction)
    createBoxFromCorners({clipped_x1, clipped_y1, oldMax.z()}, {clipped_x2, clipped_y2, newMax.z()});
    
    // Back box (negative Z direction)
    createBoxFromCorners({clipped_x1, clipped_y1, newMin.z()}, {clipped_x2, clipped_y2, oldMin.z()});

    return resultBoxes;
}

namespace{
bool areEqual(double a, double b) {
    return std::fabs(a - b) < 0.1;
}
}

std::optional<BoundingBox> GraphUpdater::mergeBoxes(const BoundingBox& box1, const BoundingBox& box2) {
    // --- A helper lambda to create sub-boxes from their min/max corners ---
    auto createBoxFromCorners = [&](const Eigen::Vector3f& minCorner, const Eigen::Vector3f& maxCorner) {
        Eigen::Vector3f dims = maxCorner - minCorner;
        Eigen::Vector3f center = minCorner + dims / 2.0f;
        BoundingBox box(dims, center);
        return box;
    };

    int coincidentFaces = 0;
    Eigen::Vector3f min1 = box1.minCorner();
    Eigen::Vector3f max1 = box1.maxCorner();
    Eigen::Vector3f min2 = box2.minCorner();
    Eigen::Vector3f max2 = box2.maxCorner();

    // A lambda to check if the face areas are aligned on a given plane
    auto facesAreAligned = [&](char axis) {
        switch (axis) {
            case 'x':
                return areEqual(min1.y(), min2.y()) && areEqual(max1.y(), max2.y()) &&
                      areEqual(min1.z(), min2.z()) && areEqual(max1.z(), max2.z());
            case 'y':
                return areEqual(min1.x(), min2.x()) && areEqual(max1.x(), max2.x()) &&
                      areEqual(min1.z(), min2.z()) && areEqual(max1.z(), max2.z());
            case 'z':
                return areEqual(min1.x(), min2.x()) && areEqual(max1.x(), max2.x()) &&
                      areEqual(min1.y(), min2.y()) && areEqual(max1.y(), max2.y());
            default:
                return false;
        }
    };

    // Check for coincident faces on the X-axis
    if (areEqual(max1.x(), min2.x()) && facesAreAligned('x')) {
        coincidentFaces++;
    }
    if (areEqual(max2.x(), min1.x()) && facesAreAligned('x')) {
        coincidentFaces++;
    }

    // Check for coincident faces on the Y-axis
    if (areEqual(max1.y(), min2.y()) && facesAreAligned('y')) {
        coincidentFaces++;
    }
    if (areEqual(max2.y(), min1.y()) && facesAreAligned('y')) {
        coincidentFaces++;
    }

    // Check for coincident faces on the Z-axis
    if (areEqual(max1.z(), min2.z()) && facesAreAligned('z')) {
        coincidentFaces++;
    }
    if (areEqual(max2.z(), min1.z()) && facesAreAligned('z')) {
        coincidentFaces++;
    }
    // VLOG(0) << "Coincident faces: " << coincidentFaces << " for boxes: "
    //         << box1.minCorner().transpose() << " - " << box1.maxCorner().transpose() << " and "
    //         << box2.minCorner().transpose() << " - " << box2.maxCorner().transpose();

    // If exactly one pair of faces are coincident, merge them.
    if (coincidentFaces == 1) {
        BoundingBox mergedBox;
        Eigen::Vector3f mergedMin = min1.cwiseMin(min2);
        Eigen::Vector3f mergedMax = max1.cwiseMax(max2);
        mergedBox = createBoxFromCorners(mergedMin, mergedMax);
        return mergedBox;
    }

    // Otherwise, the boxes cannot be merged under the specified rule.
    return std::nullopt;
}


}  // namespace hydra
