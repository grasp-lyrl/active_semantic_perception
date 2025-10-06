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

#include "Semantic/Door.h"

namespace ORB_SLAM3
{
    Door::Door() {}
    Door::~Door() {}

    int Door::getId() const
    {
        return id;
    }

    void Door::setId(int value)
    {
        id = value;
    }

    int Door::getOpId() const
    {
        return opId;
    }

    void Door::setOpId(int value)
    {
        opId = value;
    }

    int Door::getOpIdG() const
    {
        return opIdG;
    }

    void Door::setOpIdG(int value)
    {
        opIdG = value;
    }

    int Door::getMarkerId() const
    {
        return markerId;
    }

    void Door::setMarkerId(int value)
    {
        markerId = value;
    }

    std::string Door::getName() const
    {
        return name;
    }

    void Door::setName(std::string value)
    {
        name = value;
    }

    Marker *Door::getMarker() const
    {
        return marker;
    }

    void Door::setMarker(Marker *value)
    {
        marker = value;
    }

    Sophus::SE3f Door::getLocalPose() const
    {
        return localPose;
    }

    void Door::setLocalPose(const Sophus::SE3f &value)
    {
        localPose = value;
    }

    Sophus::SE3f Door::getGlobalPose() const
    {
        return globalPose;
    }

    void Door::setGlobalPose(const Sophus::SE3f &value)
    {
        globalPose = value;
    }

    Map *Door::getMap()
    {
        unique_lock<mutex> lock(mMutexMap);
        return mpMap;
    }

    void Door::setMap(Map *pMap)
    {
        unique_lock<mutex> lock(mMutexMap);
        mpMap = pMap;
    }
}