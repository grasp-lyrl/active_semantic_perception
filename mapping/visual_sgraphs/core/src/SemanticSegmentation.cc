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

#include "SemanticSegmentation.h"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/mat.hpp>

namespace ORB_SLAM3
{
    SemanticSegmentation::SemanticSegmentation(Atlas *pAtlas)
    {
        mpAtlas = pAtlas;

        // Get the system parameters
        sysParams = SystemParams::GetParams();

        // Set booleans according to the mode of operation
        mGeoRuns = !(sysParams->general.mode_of_operation == SystemParams::general::ModeOfOperation::SEM);
    }

    void SemanticSegmentation::Run()
    {
        while (true)
        {
            // Check if there are new KeyFrames in the buffer
            if (segmentedImageBuffer.empty())
            {
                usleep(3000);
                continue;
            }

            // retrieve the oldest one
            mMutexNewKFs.lock();
            std::tuple<pcl::PointCloud<pcl::PointXYZRGB>::Ptr, cv::Mat, pcl::PCLPointCloud2::Ptr, tf2::Transform, semantic_inference_msgs::DetectionResults::ConstPtr> segImgTuple = segmentedImageBuffer.front();
            segmentedImageBuffer.pop_front();
            mMutexNewKFs.unlock();

            // get the point cloud from the respective keyframe via the atlas - ignore it if KF doesn't exist
            // KeyFrame *thisKF = mpAtlas->GetKeyFrameById(std::get<0>(segImgTuple));
            // if (thisKF == nullptr || thisKF->isBad())
            //     continue;
            // const pcl::PointCloud<pcl::PointXYZRGB>::Ptr thisKFPointCloud = thisKF->getCurrentFramePointCloud();
            // if (thisKFPointCloud == nullptr)
            // {
            //     std::cout << "SemSeg: skipping KF ID: " << thisKF->mnId << ". Missing pointcloud..." << std::endl;
            //     exit(1);
            //     continue;
            // }
            const pcl::PointCloud<pcl::PointXYZRGB>::Ptr thisKFPointCloud = std::get<0>(segImgTuple);

            // separate point clouds while applying threshold
            pcl::PCLPointCloud2::Ptr pclPc2SegPrb = std::get<2>(segImgTuple);
            cv::Mat segImgUncertainity = std::get<1>(segImgTuple);
            std::vector<pcl::PointCloud<pcl::PointXYZRGBA>::Ptr> clsCloudPtrs;
            semantic_inference_msgs::DetectionResults::ConstPtr detectionResults = std::get<4>(segImgTuple);
            threshSeparatePointCloud(pclPc2SegPrb, segImgUncertainity, clsCloudPtrs, thisKFPointCloud, detectionResults);

            // get all planes for each class specific point cloud using RANSAC
            std::vector<std::vector<std::pair<pcl::PointCloud<pcl::PointXYZRGBA>::Ptr, Eigen::Vector4d>>> clsPlanes =
                getPlanesFromClassClouds(clsCloudPtrs);

            tf2::Transform tf_cTw = std::get<3>(segImgTuple);
            // Add the planes to Atlas
            updatePlaneData(tf_cTw, clsPlanes);
        }
    }

    void SemanticSegmentation::AddSegmentedFrameToBuffer(std::shared_ptr<std::tuple<pcl::PointCloud<pcl::PointXYZRGB>::Ptr, cv::Mat, pcl::PCLPointCloud2::Ptr, tf2::Transform, semantic_inference_msgs::DetectionResults::ConstPtr>> tuple)
    {
        unique_lock<std::mutex> lock(mMutexNewKFs);
        segmentedImageBuffer.push_back(*tuple);
    }

    std::list<std::tuple<pcl::PointCloud<pcl::PointXYZRGB>::Ptr, cv::Mat, pcl::PCLPointCloud2::Ptr, tf2::Transform, semantic_inference_msgs::DetectionResults::ConstPtr>> SemanticSegmentation::GetSegmentedFrameBuffer()
    {
        return segmentedImageBuffer;
    }

    void SemanticSegmentation::threshSeparatePointCloud(pcl::PCLPointCloud2::Ptr pclPc2SegPrb, cv::Mat &segImgUncertainity,
                                                        std::vector<pcl::PointCloud<pcl::PointXYZRGBA>::Ptr> &clsCloudPtrs,
                                                        const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &thisKFPointCloud,
                                                        const semantic_inference_msgs::DetectionResults::ConstPtr &detectionResults)
    {
        // parse the PointCloud2 message
        const int width = pclPc2SegPrb->width;
        const int numPoints = width * pclPc2SegPrb->height;
        const int pointStep = pclPc2SegPrb->point_step;
        const int numClasses = pointStep / bytesPerClassProb;
        const float distanceThreshNear = sysParams->pointcloud.distance_thresh.first;
        const float distanceThreshFar = sysParams->pointcloud.distance_thresh.second;
        const uint8_t confidenceThresh = sysParams->sem_seg.conf_thresh * 255;
        const float probThresh = sysParams->sem_seg.prob_thresh;

        for (int i = 0; i < numClasses; i++)
        {
            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
            pointCloud->is_dense = false;
            pointCloud->height = 1;
            clsCloudPtrs.push_back(pointCloud);
        }

        std::vector<cv::Mat> detection_masks;
        for (const auto& detected_object : detectionResults->detections)
        {
            // Convert each mask just one time
            cv::Mat mask = cv_bridge::toCvCopy(detected_object.mask, sensor_msgs::image_encodings::TYPE_32FC1)->image;

            if (mask.cols == width && mask.rows == pclPc2SegPrb->height)
            {
                detection_masks.push_back(mask);
            }
            else
            {
                std::cout << "Mask dimensions do not match point cloud dimensions." << std::endl;
            }
        }

        // apply thresholding and track confidence (complement of uncertainty)
        const uint8_t *data = pclPc2SegPrb->data.data();
        for (int j = 0; j < numClasses; j++)
        {
            for (int i = 0; i < numPoints; i++)
            {
                float value;
                memcpy(&value, data + pointStep * i + bytesPerClassProb * j + pclPc2SegPrb->fields[0].offset, bytesPerClassProb);

                if (value >= probThresh)
                {
                    // inject coordinates as a point to respective point cloud
                    pcl::PointXYZRGBA point;
                    point.y = static_cast<int>(i / width);
                    point.x = i % width;

                    // get the original point from the keyframe point cloud
                    const pcl::PointXYZRGB origPoint = thisKFPointCloud->at(point.x, point.y);
                    if (!pcl::isFinite(origPoint))
                        continue;

                    bool valid = true;
                    for (const auto& mask : detection_masks) // Loop through the fast, pre-computed masks
                    {
                        if (mask.at<float>(point.y, point.x) > 0)
                        {
                            valid = false;
                            break;
                        }
                    }
                    if (!valid)
                        continue;

                    // convert uncertainity to single value and assign confidence to alpha channel
                    cv::Vec3b vec = segImgUncertainity.at<cv::Vec3b>(point.y, point.x);
                    point.a = 255 - static_cast<int>(0.299 * vec[2] + 0.587 * vec[1] + 0.114 * vec[0]);

                    // exclude points with low confidence
                    if (point.a < confidenceThresh)
                        continue;

                    // assign the XYZ and RGB values to the surviving point before pushing to specific point cloud
                    point.x = origPoint.x;
                    point.y = origPoint.y;
                    point.z = origPoint.z;
                    point.r = origPoint.r;
                    point.g = origPoint.g;
                    point.b = origPoint.b;

                    // confidence as the squared inverse depth - interpolated between near and far thresholds
                    // confidence = 255 for near, 45 for far, and interpolated according to squared distance
                    const float thresholdNear = sysParams->pointcloud.distance_thresh.first;
                    const float thresholdFar = sysParams->pointcloud.distance_thresh.second;
                    if (point.z < thresholdNear)
                        point.a = 255;
                    else if (point.z > thresholdFar)
                        point.a = 45;
                    else
                        point.a = 255 - static_cast<int>(210 * sqrt((point.z - thresholdNear) / (thresholdFar - thresholdNear)));

                    // add the point to the respective class specific point cloud
                    clsCloudPtrs[j]->push_back(point);
                }
            }
        }

        // specify size/width and header for each class specific point cloud
        for (int i = 0; i < numClasses; i++)
        {
            clsCloudPtrs[i]->width = clsCloudPtrs[i]->size();
            clsCloudPtrs[i]->header = pclPc2SegPrb->header;
        }
    }

    std::vector<std::vector<std::pair<pcl::PointCloud<pcl::PointXYZRGBA>::Ptr, Eigen::Vector4d>>> SemanticSegmentation::getPlanesFromClassClouds(
        std::vector<pcl::PointCloud<pcl::PointXYZRGBA>::Ptr> &clsCloudPtrs)
    {
        std::vector<std::vector<std::pair<pcl::PointCloud<pcl::PointXYZRGBA>::Ptr, Eigen::Vector4d>>> clsPlanes;

        // downsample/filter the pointcloud and extract planes
        for (size_t i = 0; i < clsCloudPtrs.size(); i++)
        {
            // [TODO?] - Perhaps consider points in order of confidence instead of downsampling
            // Downsample the given pointcloud after filtering based on distance
            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr filteredCloud;
            filteredCloud = Utils::pointcloudDistanceFilter<pcl::PointXYZRGBA>(clsCloudPtrs[i]);
            filteredCloud = Utils::pointcloudDownsample<pcl::PointXYZRGBA>(filteredCloud,
                                                                           sysParams->sem_seg.pointcloud.downsample.leaf_size,
                                                                           sysParams->sem_seg.pointcloud.downsample.min_points_per_voxel);
            filteredCloud = Utils::pointcloudOutlierRemoval<pcl::PointXYZRGBA>(filteredCloud,
                                                                               sysParams->sem_seg.pointcloud.outlier_removal.std_threshold,
                                                                               sysParams->sem_seg.pointcloud.outlier_removal.mean_threshold);

            // copy the filtered cloud for later storage into the keyframe
            pcl::copyPointCloud(*filteredCloud, *clsCloudPtrs[i]);

            std::vector<std::pair<pcl::PointCloud<pcl::PointXYZRGBA>::Ptr, Eigen::Vector4d>> extractedPlanes;
            if (i == ORB_SLAM3::Plane::planeVariant::WALL && filteredCloud->points.size() > sysParams->seg.wall_pointclouds_thresh)
            {
                extractedPlanes = Utils::ransacPlaneFitting<pcl::PointXYZRGBA, pcl::WeightedSACSegmentation>(filteredCloud, sysParams->seg.wall_pointclouds_thresh);
            }
            else if (i == ORB_SLAM3::Plane::planeVariant::DOOR && filteredCloud->points.size() > sysParams->seg.door_pointclouds_thresh)
            {
                extractedPlanes = Utils::ransacPlaneFitting<pcl::PointXYZRGBA, pcl::WeightedSACSegmentation>(filteredCloud, sysParams->seg.door_pointclouds_thresh);
            }
            else
            {
                extractedPlanes = Utils::ransacPlaneFitting<pcl::PointXYZRGBA, pcl::SACSegmentation>(filteredCloud, sysParams->seg.pointclouds_thresh);
            }
            clsPlanes.push_back(extractedPlanes);
        }
        return clsPlanes;
    }

    void SemanticSegmentation::updatePlaneData(tf2::Transform& tf_cTw, std::vector<std::vector<std::pair<pcl::PointCloud<pcl::PointXYZRGBA>::Ptr,
                                                                                 Eigen::Vector4d>>> &clsPlanes)
    {
        for (size_t clsId = 0; clsId < clsPlanes.size(); clsId++)
        {
            for (const auto &planePoint : clsPlanes[clsId])
            {
                // Get the plane equation
                Eigen::Vector4d estimatedPlane = planePoint.second;
                if (estimatedPlane ==  Eigen::Vector4d::Zero())
                {
                    continue; // Skip if the plane is not detected
                }
                g2o::Plane3D detectedPlane(estimatedPlane);
                geometry_msgs::Transform geom_transform = tf2::toMsg(tf_cTw);
                Eigen::Matrix4d eigen_cTw = tf2::transformToEigen(geom_transform).matrix().cast<double>();
                Eigen::Matrix4d eigen_wTc = tf2::transformToEigen(geom_transform).inverse().matrix().cast<double>();

                // Convert the given plane to global coordinates
                g2o::Plane3D globalEquation = Utils::applyPoseToPlane(eigen_cTw,
                                                                      detectedPlane);

                // Compute the average confidence across all pixels in the plane observation
                pcl::PointCloud<pcl::PointXYZRGBA>::Ptr planeCloud = planePoint.first;
                std::vector<double> confidences;
                for (size_t i = 0; i < planeCloud->size(); i++)
                    confidences.push_back(static_cast<int>(planeCloud->points[i].a) / 255.0);
                // [Note] - use softmin when dealing with semantic confidences
                // double conf = Utils::calcSoftMin(confidences);
                // [Note] - use average when dealing with geometric (in this case depth) confidences
                double conf = std::accumulate(confidences.begin(), confidences.end(), 0.0) / confidences.size();

                // temp global plane cloud
                pcl::PointCloud<pcl::PointXYZRGBA>::Ptr globalPlaneCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
                pcl::copyPointCloud(*planeCloud, *globalPlaneCloud);
                pcl::transformPointCloud(*globalPlaneCloud, *globalPlaneCloud, eigen_cTw);

                // Get the semantic type of the observation
                ORB_SLAM3::Plane::planeVariant semanticType = Utils::getPlaneTypeFromClassId(clsId);

                // Check if we need to add the wall to the map or not
                int matchedPlaneId = Utils::associatePlanes(mpAtlas->GetAllPlanes(),
                                                            detectedPlane,
                                                            globalPlaneCloud,
                                                            eigen_wTc,
                                                            semanticType,
                                                            sysParams->seg.plane_association.ominus_thresh);

                if (matchedPlaneId == -1)
                {
                    if (!mGeoRuns)
                    {
                        ORB_SLAM3::Plane *newMapPlane = GeoSemHelpers::createMapPlane(mpAtlas, eigen_cTw, eigen_wTc, detectedPlane,
                                                                                      planeCloud, semanticType, conf);
                        // Cast a vote for the plane semantics
                        updatePlaneSemantics(newMapPlane->getId(), clsId, conf);
                    }
                }
                else
                {
                    if (!mGeoRuns)
                        GeoSemHelpers::updateMapPlane(mpAtlas, eigen_cTw, eigen_wTc, detectedPlane, planeCloud,
                                                      matchedPlaneId, semanticType, conf);
                    else
                    {
                        pcl::transformPointCloud(*planeCloud, *planeCloud, eigen_cTw);
                        ORB_SLAM3::Plane *matchedPlane = mpAtlas->GetPlaneById(matchedPlaneId);
                        // Add the plane cloud to the matched plane
                        if (!planeCloud->empty())
                            matchedPlane->setMapClouds(planeCloud);
                    }

                    // Cast a vote for the plane semantics
                    updatePlaneSemantics(matchedPlaneId, clsId, conf);
                }
            }
        }
    }

    void SemanticSegmentation::updatePlaneSemantics(int planeId, int clsId, double confidence)
    {
        // retrieve the plane from the map
        Plane *matchedPlane = mpAtlas->GetPlaneById(planeId);

        // plane type compatible with the Plane class
        ORB_SLAM3::Plane::planeVariant planeType = Utils::getPlaneTypeFromClassId(clsId);

        // cast a vote for the plane semantics
        matchedPlane->castWeightedVote(planeType, confidence);
    }
}