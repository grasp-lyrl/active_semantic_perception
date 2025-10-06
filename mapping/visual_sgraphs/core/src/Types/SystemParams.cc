/**
 * This file is part of Visual S-Graphs (vS-Graphs).
 * Copyright (C) 2023-2025 SnT, University of Luxembourg
 *
 * 📝 Authors: Ali Tourani, Saad Ejaz, Hriday Bavle, Jose Luis Sanchez-Lopez, and Holger Voos
 *
 * vS-Graphs is free software: you can redistribute it and/or modify it under the terms
 * of the GNU General Public License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later version.
 *
 * This software is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details: https://www.gnu.org/licenses/
*/

#include "Types/SystemParams.h"

namespace ORB_SLAM3
{
    SystemParams *SystemParams::mSystemParams = nullptr;

    SystemParams::SystemParams()
    {
        mSystemParams = nullptr;
    }

    SystemParams *SystemParams::GetParams()
    {
        if (mSystemParams == nullptr)
            mSystemParams = new SystemParams();
        return mSystemParams;
    }

    void SystemParams::SetParams(const std::string &strConfigFile)
    {
        std::cout << "- Loading system parameters from " << strConfigFile << std::endl;
        try
        {
            mConfig = YAML::LoadFile(strConfigFile);
            std::cout << "- System parameters loaded!\n\n";
        }
        catch (YAML::BadFile &e)
        {
            std::cout << "- Error loading configuration file " << e.what() << std::endl;
            std::cout << "- Exiting ... \n\n";
            exit(1);
        }

        // Set parameters
        try
        {
            // General Parameters
            general.env_database = mConfig["general"]["env_database"].as<std::string>();
            general.mode_of_operation = static_cast<general::ModeOfOperation>(mConfig["general"]["mode_of_operation"].as<int>());

            // Marker Parameters
            markers.impact = mConfig["markers"]["impact"].as<float>();

            // Tracking Refinement Parameters
            refine_map_points.enabled = mConfig["refine_map_points"]["enabled"].as<bool>();
            refine_map_points.max_distance_for_delete = mConfig["refine_map_points"]["max_distance_for_delete"].as<float>();
            refine_map_points.octree.resolution = mConfig["refine_map_points"]["octree"]["resolution"].as<float>();
            refine_map_points.octree.search_radius = mConfig["refine_map_points"]["octree"]["search_radius"].as<float>();
            refine_map_points.octree.min_neighbors = mConfig["refine_map_points"]["octree"]["min_neighbors"].as<unsigned int>();

            // Plane based Covisibility Parameters
            plane_based_covisibility.enabled = mConfig["plane_based_covisibility"]["enabled"].as<bool>();
            plane_based_covisibility.max_keyframes = mConfig["plane_based_covisibility"]["max_keyframes"].as<unsigned int>();
            plane_based_covisibility.score_per_plane = mConfig["plane_based_covisibility"]["score_per_plane"].as<unsigned int>();

            // Common Segmentation Parameters
            seg.wall_pointclouds_thresh = mConfig["seg"]["wall_pointclouds_thresh"].as<unsigned int>();
            seg.pointclouds_thresh = mConfig["seg"]["pointclouds_thresh"].as<unsigned int>();
            seg.door_pointclouds_thresh = mConfig["seg"]["door_pointclouds_thresh"].as<unsigned int>();
            seg.plane_point_dist_thresh = mConfig["seg"]["plane_point_dist_thresh"].as<float>();
            seg.plane_association.ominus_thresh = mConfig["seg"]["plane_association"]["ominus_thresh"].as<float>();
            seg.plane_association.distance_thresh = mConfig["seg"]["plane_association"]["distance_thresh"].as<float>();
            seg.plane_association.centroid_thresh = mConfig["seg"]["plane_association"]["centroid_thresh"].as<float>();
            seg.plane_association.cluster_separation.enabled = mConfig["seg"]["plane_association"]["cluster_separation"]["enabled"].as<bool>();
            seg.plane_association.cluster_separation.tolerance = mConfig["seg"]["plane_association"]["cluster_separation"]["tolerance"].as<float>();
            seg.plane_association.cluster_separation.downsample.leaf_size = mConfig["seg"]["plane_association"]["cluster_separation"]["downsample"]["leaf_size"].as<float>();
            seg.plane_association.cluster_separation.downsample.min_points_per_voxel = mConfig["seg"]["plane_association"]["cluster_separation"]["downsample"]["min_points_per_voxel"].as<unsigned int>();
            seg.ransac.max_planes = mConfig["seg"]["ransac"]["max_planes"].as<unsigned int>();
            seg.ransac.distance_thresh = mConfig["seg"]["ransac"]["distance_thresh"].as<float>();
            seg.ransac.max_iterations = mConfig["seg"]["ransac"]["max_iterations"].as<unsigned int>();

            // Optimization Parameters
            optimization.marginalize_planes = mConfig["optimization"]["marginalize_planes"].as<bool>();
            optimization.plane_kf.enabled = mConfig["optimization"]["plane_kf"]["enabled"].as<bool>();
            optimization.plane_kf.information_gain = mConfig["optimization"]["plane_kf"]["information_gain"].as<float>();
            optimization.plane_point.enabled = mConfig["optimization"]["plane_point"]["enabled"].as<bool>();
            optimization.plane_point.information_gain = mConfig["optimization"]["plane_point"]["information_gain"].as<float>();
            optimization.plane_map_point.enabled = mConfig["optimization"]["plane_map_point"]["enabled"].as<bool>();
            optimization.plane_map_point.information_gain = mConfig["optimization"]["plane_map_point"]["information_gain"].as<float>();

            // Geometric Segmentation Parameters
            geo_seg.pointcloud.downsample.leaf_size = mConfig["geo_seg"]["pointcloud"]["downsample"]["leaf_size"].as<float>();
            geo_seg.pointcloud.downsample.min_points_per_voxel = mConfig["geo_seg"]["pointcloud"]["downsample"]["min_points_per_voxel"].as<unsigned int>();
            geo_seg.pointcloud.outlier_removal.std_threshold = mConfig["geo_seg"]["pointcloud"]["outlier_removal"]["std_threshold"].as<float>();
            geo_seg.pointcloud.outlier_removal.mean_threshold = mConfig["geo_seg"]["pointcloud"]["outlier_removal"]["mean_threshold"].as<unsigned int>();

            // Semantic Segmentation Parameters
            sem_seg.min_votes = mConfig["sem_seg"]["min_votes"].as<float>();
            sem_seg.prob_thresh = mConfig["sem_seg"]["prob_thresh"].as<float>();
            sem_seg.conf_thresh = mConfig["sem_seg"]["conf_thresh"].as<float>();
            sem_seg.max_tilt_wall = mConfig["sem_seg"]["max_tilt_wall"].as<float>();
            sem_seg.max_tilt_ground = mConfig["sem_seg"]["max_tilt_ground"].as<float>();
            sem_seg.max_step_elevation = mConfig["sem_seg"]["max_step_elevation"].as<float>();
            sem_seg.pointcloud.downsample.leaf_size = mConfig["sem_seg"]["pointcloud"]["downsample"]["leaf_size"].as<float>();
            sem_seg.pointcloud.downsample.min_points_per_voxel = mConfig["sem_seg"]["pointcloud"]["downsample"]["min_points_per_voxel"].as<unsigned int>();
            sem_seg.pointcloud.outlier_removal.std_threshold = mConfig["sem_seg"]["pointcloud"]["outlier_removal"]["std_threshold"].as<float>();
            sem_seg.pointcloud.outlier_removal.mean_threshold = mConfig["sem_seg"]["pointcloud"]["outlier_removal"]["mean_threshold"].as<unsigned int>();
            sem_seg.reassociate.enabled = mConfig["sem_seg"]["reassociate"]["enabled"].as<bool>();
            sem_seg.reassociate.association_thresh = mConfig["sem_seg"]["reassociate"]["association_thresh"].as<float>();

            // Room Segmentation Parameters
            room_seg.method = static_cast<room_seg::Method>(mConfig["room_seg"]["method"].as<int>());
            room_seg.center_distance_thresh = mConfig["room_seg"]["center_distance_thresh"].as<float>();
            room_seg.plane_facing_dot_thresh = mConfig["room_seg"]["plane_facing_dot_thresh"].as<float>();
            room_seg.min_wall_distance_thresh = mConfig["room_seg"]["min_wall_distance_thresh"].as<float>();
            room_seg.walls_perpendicularity_thresh = mConfig["room_seg"]["perpendicularity_thresh"].as<float>();
            room_seg.min_cluster_vertices = mConfig["room_seg"]["skeleton_based"]["min_cluster_vertices"].as<unsigned int>();
            room_seg.marker_wall_distance_thresh = mConfig["room_seg"]["geo_based"]["marker_wall_distance_thresh"].as<float>();
            room_seg.cluster_point_wall_distance_thresh = mConfig["room_seg"]["skeleton_based"]["cluster_point_wall_distance_thresh"].as<float>();
        }
        catch (YAML::Exception &e)
        {
            std::cerr << "Error loading system parameters. Make sure all parameters are defined properly: " << e.what() << std::endl;
            exit(1);
        }
    }
}
