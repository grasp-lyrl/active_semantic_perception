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
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseStamped
import yaml

# Load the configurations
config_file = open("config.yaml", "r")
config = yaml.load(config_file, Loader=yaml.FullLoader)

files_path = config["results_dir"]
slam_method = config["slam_method"]
dataset_seq = config["dataset_seq"]
gt_pose_topic = config["ros_topics"]["ground_truth_pose"]
slam_pose_topic = config["ros_topics"]["camera_pose"]

if len(sys.argv) > 1:
    # use that as identifier
    dataset_seq += sys.argv[1]

# Creating a txt file that will contain poses
print("Creating txt files for adding SLAM and Ground-Truth poses ...")
gt_pose_file = open(
    f"{files_path}/gt_pose_{slam_method}_{dataset_seq}.txt", "w+")
gt_pose_file.write("#timestamp tx ty tz qx qy qz qw\n")
gt_pose_file.flush()
slam_pose_file = open(
    f"{files_path}/slam_pose_{slam_method}_{dataset_seq}.txt", "w+")
slam_pose_file.write("#timestamp tx ty tz qx qy qz qw\n")
slam_pose_file.flush()


def groundtruthPoseCallback(groundtruth_pose_msg):
    # Calculate different variables
    time = rospy.get_rostime().to_sec()
    tx = groundtruth_pose_msg.pose.position.x
    ty = groundtruth_pose_msg.pose.position.y
    tz = groundtruth_pose_msg.pose.position.z
    rx = groundtruth_pose_msg.pose.orientation.x
    ry = groundtruth_pose_msg.pose.orientation.y
    rz = groundtruth_pose_msg.pose.orientation.z
    rw = groundtruth_pose_msg.pose.orientation.w
    # Write to the Ground-Truth file
    gt_pose_file.write(
        str(time) + " " + str(tx) + " " + str(ty) + " "
        + str(tz) + " " + str(rx) + " " + str(ry) + " "
        + str(rz) + " " + str(rw) + "\n"
    )
    gt_pose_file.flush()


def slamPoseCallback(slam_pose_msg):
    # Calculate different variables
    time = slam_pose_msg.header.stamp.to_sec()
    odom_x = slam_pose_msg.pose.position.x
    odom_y = slam_pose_msg.pose.position.y
    odom_z = slam_pose_msg.pose.position.z
    odom_rx = slam_pose_msg.pose.orientation.x
    odom_ry = slam_pose_msg.pose.orientation.y
    odom_rz = slam_pose_msg.pose.orientation.z
    odom_rw = slam_pose_msg.pose.orientation.w
    # Write to the SLAM file
    slam_pose_file.write(
        str(time) + " " + str(odom_x) + " " + str(odom_y) + " "
        + str(odom_z) + " " + str(odom_rx) + " " + str(odom_ry) + " "
        + str(odom_rz) + " " + str(odom_rw) + "\n"
    )
    slam_pose_file.flush()


def subscribers():
    rospy.init_node("text_file_generator", anonymous=True)
    # Subscriber to the ground-truth topic
    rospy.Subscriber(gt_pose_topic,
                     ModelStates, groundtruthPoseCallback)
    # Subscriber to the SLAM topic
    rospy.Subscriber(slam_pose_topic, PoseStamped, slamPoseCallback)
    rospy.spin()


if __name__ == "__main__":
    subscribers()
