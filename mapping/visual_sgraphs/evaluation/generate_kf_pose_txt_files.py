#!/usr/bin/env python3

'''
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
'''

import sys
import rospy
from nav_msgs.msg import Path
import yaml

# Load the configurations
config_file = open("config.yaml", "r")
config = yaml.load(config_file, Loader=yaml.FullLoader)

files_path = config["results_dir"]
slam_method = config["slam_method"]
dataset_seq = config["dataset_seq"]
slam_pose_topic = config["ros_topics"]["keyframe_list"]

if len(sys.argv) > 1:
    # use that as identifier
    dataset_seq += sys.argv[1]

# Creating a txt file that will contain poses
print("Creating txt file for adding SLAM poses ...")
slam_pose_file_path = f"{files_path}/slam_pose_{slam_method}_{dataset_seq}.txt"

def write_pose_file(file_path, poses):
    with open(file_path, "w") as pose_file:
        pose_file.write("#timestamp tx ty tz qx qy qz qw\n")
        for pose in poses:
            time = pose.header.stamp.to_sec()
            tx = pose.pose.position.x
            ty = pose.pose.position.y
            tz = pose.pose.position.z
            rx = pose.pose.orientation.x
            ry = pose.pose.orientation.y
            rz = pose.pose.orientation.z
            rw = pose.pose.orientation.w
            pose_file.write(f"{time} {tx} {ty} {tz} {rx} {ry} {rz} {rw}\n")


def slamPoseCallback(slam_path_msg):
    poses = slam_path_msg.poses
    write_pose_file(slam_pose_file_path, poses)


def subscribers():
    rospy.init_node("text_file_generator", anonymous=True)
    # Subscriber to the SLAM topic
    rospy.Subscriber(slam_pose_topic, Path, slamPoseCallback)
    rospy.spin()


if __name__ == "__main__":
    subscribers()