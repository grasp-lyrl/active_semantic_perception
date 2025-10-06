#pragma once
#include "hydra/common/dsg_types.h"
#include "hydra/frontend/view_selector.h"
#include <opencv2/core/mat.hpp>

namespace hydra {

struct InputData;

class ViewDatabase {
 public:
  using Ptr = std::shared_ptr<ViewDatabase>;
  struct Config {
    std::string view_selection_method = "boundary";
  };

  explicit ViewDatabase(const Config& config, const int scene_graph_id = 0);

  ~ViewDatabase();

  void updateAssignments(const DynamicSceneGraph& graph,
                         const std::unordered_set<NodeId>& active_places) const;

 protected:
  mutable ViewSelector::FeatureList views_;
  std::unique_ptr<ViewSelector> view_selector_;
  int scene_graph_id_;
};

void declare_config(ViewDatabase::Config& config);

}  // namespace hydra
