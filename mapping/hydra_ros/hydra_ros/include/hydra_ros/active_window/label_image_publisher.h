#pragma once
#include <hydra/active_window/reconstruction_module.h>
#include <ros/ros.h>

namespace hydra {


class LabelImagePublisher : public ReconstructionModule::Sink{
 public:
    struct Config{
        std::string ns = "~";
    } const config;

    explicit LabelImagePublisher(const Config& config);

    virtual ~LabelImagePublisher();

    void call(uint64_t timestamp_ns,
                const VolumetricMap& map,
                ActiveWindowOutput& msg) const override;

 private:
    ros::NodeHandle nh_;
    ros::Publisher publisher_;
    void publish(uint64_t timestamp_ns, const ActiveWindowOutput data) const;

};

void declare_config(LabelImagePublisher::Config& config);

}  // namespace hydra
