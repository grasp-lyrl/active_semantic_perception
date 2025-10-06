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

#ifndef ROOM_H
#define ROOM_H

#include "Door.h"
#include "Geometric/Plane.h"
#include "Thirdparty/g2o/g2o/types/vertex_plane.h"

namespace ORB_SLAM3
{

    class Room
    {
    private:
        int id;                         // The room's identifier
        int opId;                       // The room's identifier in the local optimizer
        int opIdG;                      // The room's identifier in the global optimizer
        bool isCorridor;                // Checks if the room is a corridor or not
        int metaMarkerId;               // The identifier of the room's meta-marker (containing information about the room)
        std::string name;               // The name devoted for each room (optional)
        bool hasKnownLabel;             // Checks if it is a candidate room (meta-marker detected) or not
        Marker *metaMarker;             // The meta-marker assigned for the room
        Plane *groundPlane;             // The ground plane associated with the room
        std::vector<Door *> doors;      // The vector of detected doors of a room
        std::vector<Plane *> walls;     // The vector of detected walls of a room
        Eigen::Vector3d roomCenter;     // The center of the room as a 3D vector in the global reference
        std::vector<int> doorMarkerIds; // Markers attached to the doors of a room [in real map], e.g. [3, 4]

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        Room();
        ~Room();

        int getId() const;
        void setId(int value);

        int getOpId() const;
        void setOpId(int value);

        int getOpIdG() const;
        void setOpIdG(int value);

        bool getIsCorridor() const;
        void setIsCorridor(bool value);

        bool getHasKnownLabel() const;
        void setHasKnownLabel(bool value);

        int getMetaMarkerId() const;
        void setMetaMarkerId(int value);

        Marker *getMetaMarker() const;
        void setMetaMarker(Marker *value);

        std::string getName() const;
        void setName(std::string value);

        void setDoors(Door *value);
        std::vector<Door *> getDoors() const;

        void setWalls(Plane *value);
        std::vector<Plane *> getWalls() const;

        void clearWalls();

        Plane *getGroundPlane() const;
        void setGroundPlane(Plane *ground);

        void setDoorMarkerIds(int value);
        std::vector<int> getDoorMarkerIds() const;

        Eigen::Vector3d getRoomCenter() const;
        void setRoomCenter(Eigen::Vector3d value);

        Map *getMap();
        void setMap(Map *pMap);

    protected:
        Map *mpMap;
        std::mutex mMutexMap;
    };
}

#endif