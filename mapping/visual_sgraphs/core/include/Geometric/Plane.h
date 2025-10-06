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

#ifndef PLANE_H
#define PLANE_H

#include <set>
#include "Map.h"
#include "MapPoint.h"
#include "Semantic/Marker.h"
#include "Types/SystemParams.h"
#include "Thirdparty/g2o/g2o/types/plane3d.h"

#include <pcl/common/io.h>
#include <pcl/common/centroid.h>
#include <pcl/octree/octree_search.h>

namespace ORB_SLAM3
{
    class Map;
    class Marker;
    class MapPoint;

    class Plane
    {
    public:
        enum planeVariant
        {
            UNDEFINED = -1,
            CURTAIN = 0,
            BLIND = 1,
            WINDOW = 2,
            WALL = 3,
            DOOR = 4,
            GROUND = 5
        };

        bool excludedFromAssoc; // The plane's exclusion from association (once excluded, can't be associated again)

        struct Observation
        {
            g2o::Plane3D localPlane;                        // The plane equation in the local frame
            Eigen::Matrix4d Gij;                            // The aggregated point cloud measurement for point-plane constraint
            double confidence;                              // The aggregated confidence of the plane
            planeVariant semanticType = UNDEFINED;          // The semantic type of the plane
        };

        // Variables for bundle adjustment
        KeyFrame *refKeyFrame;             // The first keyframe that observed the plane is the reference keyframe
        unsigned long int mnBAGlobalForKF; // The reference keyframe ID for the Global BA the plane was part of
        g2o::Plane3D mPlaneGBA;            // The plane equation in the global map after the Global BA

    private:
        int id;                                             // The plane's identifier
        int opId;                                           // The plane's identifier in the local optimizer
        int opIdG;                                          // The plane's identifier in the global optimizer
        planeVariant planeType;                             // The plane's semantic type (e.g., wall, ground, etc.)
        Eigen::Vector3f centroid;                           // The centroid of the plane
        std::vector<uint8_t> color;                         // A color devoted for visualization
        g2o::Plane3D localEquation;                         // The plane equation in the local map
        g2o::Plane3D globalEquation;                        // The plane equation in the global map
        std::set<MapPoint *> mapPoints;                     // The unique set of map points lying on the plane
        std::map<planeVariant, double> semanticVotes;       // The votes for the semantic type of the plane
        // std::map<KeyFrame *, Observation> observations;    // Plane's observations in keyFrames
        std::vector<Observation> observations;                // Plane's observations in keyFrames
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr planeCloud; // The point cloud of the plane
        boost::shared_ptr<pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGBA>> octree; // The octree for the plane cloud

    public:
        Plane();
        ~Plane();

        int getId() const;
        void setId(int value);

        int getOpId() const;
        void setOpId(int value);

        int getOpIdG() const;
        void setOpIdG(int value);

        void setColor();
        std::vector<uint8_t> getColor() const;

        planeVariant getPlaneType();
        planeVariant getExpectedPlaneType();
        void setPlaneType(planeVariant newType);

        void setMapPoints(MapPoint *value);
        std::set<MapPoint *> getMapPoints();

        Eigen::Vector3f getCentroid() const;
        void setCentroid(const Eigen::Vector3f &value);

        g2o::Plane3D getLocalEquation() const;
        void setLocalEquation(const g2o::Plane3D &value);

        g2o::Plane3D getGlobalEquation() const;
        void setGlobalEquation(const g2o::Plane3D &value);

        void addObservation(Observation obs);
        void eraseObservation(KeyFrame *pKF);
        const std::map<KeyFrame *, Plane::Observation> &getObservations() const;
        const std::vector<Observation> &getNewObservations() const;

        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr getMapClouds();
        void setMapClouds(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr value);
        void replaceMapClouds(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr value);
        bool isPointinPlaneCloud(const Eigen::Vector3d &point);

        void castWeightedVote(planeVariant semanticType, double voteWeight);
        void resetPlaneSemantics();

        Map *GetMap();
        void SetMap(Map *pMap);

    protected:
        Map *mpMap;
        std::mutex mMutexMap, mMutexType;
        mutable std::mutex mMutexFeatures, mMutexPos;
    };
}

#endif