#!/usr/bin/env python

import rospy
from sensor_msgs.msg import CameraInfo, Image

class CameraInfoPublisher:
    def __init__(self):
        """
        Initializes the node, publisher, and subscriber.
        """
        # Create a publisher for the /dominic/forward/color/camera_info topic.
        self.pub = rospy.Publisher('/dominic/forward/color/camera_info', CameraInfo, queue_size=10)

        # Create a subscriber to the /dominic/forward/color/image_raw topic.
        self.subscriber = rospy.Subscriber('/dominic/forward/color/image_raw', Image, self.image_callback)
        
        # Create a reusable CameraInfo message.
        self.cam_info_msg = self.create_camera_info()

    def create_camera_info(self):
        """
        Creates and returns a static CameraInfo message.
        """
        cam_info = CameraInfo()
        
        # Fill in the camera parameters.
        # These values are static, so we can set them once.
        cam_info.height = 480
        cam_info.width = 640
        cam_info.distortion_model = "plumb_bob"
        
        # Distortion coefficients (D)
        cam_info.D = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Intrinsic camera matrix (K)
        cam_info.K = [320.0,   0.0, 319.5, 
                      0.0, 240.0, 239.5, 
                      0.0,   0.0,   1.0]
        
        # Rectification matrix (R)
        cam_info.R = [1.0, 0.0, 0.0, 
                      0.0, 1.0, 0.0, 
                      0.0, 0.0, 1.0]
        
        # Projection/camera matrix (P)
        cam_info.P = [320.0,   0.0, 319.5, 0.0,
                      0.0, 240.0, 239.5, 0.0,
                      0.0,   0.0,   1.0, 0.0]
                      
        return cam_info

    def image_callback(self, image_msg):
        """
        Callback function that is triggered by incoming Image messages.
        """
        # Update the header information with data from the incoming image.
        self.cam_info_msg.header.stamp = image_msg.header.stamp
        self.cam_info_msg.header.frame_id = "dominic/forward_link" # Or image_msg.header.frame_id
        
        # Publish the CameraInfo message.
        self.pub.publish(self.cam_info_msg)

if __name__ == '__main__':
    try:
        # Initialize the ROS node.
        rospy.init_node('camera_info_publisher_node', anonymous=True)
        # Create an instance of our class.
        CameraInfoPublisher()
        # Keep the node running until it's shut down.
        rospy.spin()
    except rospy.ROSInterruptException:
        pass