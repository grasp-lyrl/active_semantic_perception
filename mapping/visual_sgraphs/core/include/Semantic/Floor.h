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

#ifndef FLOOR_H
#define FLOOR_H

#include "Map.h"
#include "Door.h"
#include "Room.h"
#include "Geometric/Plane.h"

namespace ORB_SLAM3
{
    class Room;

    class Floor
    {
    private:
        int id;                     // The floor's identifier
        int opId;                   // The floor's identifier in the local optimizer
        int opIdG;                  // The floor's identifier in the global optimizer
        std::vector<Door *> doors;  // The vector of detected doors in a building floor
        std::vector<Room *> rooms;  // The vector of detected rooms in a building floor
        std::vector<Plane *> walls; // The vector of detected walls in a building floor

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        Floor();
        ~Floor();

        int getId() const;
        void setId(int value);

        int getOpId() const;
        void setOpId(int value);

        int getOpIdG() const;
        void setOpIdG(int value);

        void setDoors(Door *value);
        std::vector<Door *> getDoors() const;

        void setRooms(Room *value);
        std::vector<Room *> getRooms() const;

        void setWalls(Plane *value);
        std::vector<Plane *> getWalls() const;

        Map *getMap();
        void setMap(Map *pMap);

    protected:
        Map *mpMap;
        std::mutex mMutexMap;
    };
}

#endif