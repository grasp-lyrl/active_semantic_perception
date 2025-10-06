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

#ifndef MARKER_H
#define MARKER_H

#include "Map.h"
#include "KeyFrame.h"

namespace ORB_SLAM3
{
    class Map;
    class KeyFrame;

    class Marker
    {
    public:
        enum markerVariant
        {
            UNKNOWN = -1,
            ON_DOOR = 0,
            ON_WALL = 1,
            ON_ROOM_CENTER = 2
        };

    private:
        int id;                                          // The marker's identifier
        int opId;                                        // The marker's identifier in the local optimizer
        int opIdG;                                       // The marker's identifier in the global optimizer
        double time;                                     // The timestamp (in seconds) of observing the marker
        bool markerInGMap;                               // Check if the marker is in the Global Map or not
        Sophus::SE3f localPose;                          // Marker's pose (position and orientation) in the Local Map
        Sophus::SE3f globalPose;                         // Marker's pose (position and orientation) in the Global Map
        markerVariant markerType;                        // The semantic object the marker is labeled with (e.g., wall, etc.)
        std::map<KeyFrame *, Sophus::SE3f> observations; // Marker's observations in KeyFrames

    public:
        Marker();
        ~Marker();

        int getId() const;
        void setId(int value);

        int getOpId() const;
        void setOpId(int value);

        int getOpIdG() const;
        void setOpIdG(int value);

        double getTime() const;
        void setTime(double value);

        markerVariant getMarkerType() const;
        void setMarkerType(markerVariant newType);

        bool isMarkerInGMap() const;
        void setMarkerInGMap(bool value);

        Sophus::SE3f getLocalPose() const;
        void setLocalPose(const Sophus::SE3f &value);

        Sophus::SE3f getGlobalPose() const;
        void setGlobalPose(const Sophus::SE3f &value);

        void addObservation(KeyFrame *pKF, Sophus::SE3f localPose);
        const std::map<KeyFrame *, Sophus::SE3f> &getObservations() const;

        Map *getMap();
        void setMap(Map *pMap);

    protected:
        Map *mpMap;
        std::mutex mMutexMap;
    };

}

#endif