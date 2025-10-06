#!/usr/bin/env python

import torch
import rospy
import numpy as np
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from modelRunner import yosoInit, yosoSegmenter
from utils.helpers import cleanMemory, monitorParams
from utils.semantic_utils import probabilities2ROSMsg
from output import yosoVisualizer, entropyVisualizer
from segmenter_ros.msg import SegmenterDataMsg, VSGraphDataMsg
import queue
import threading
import time


class Segmenter:
    def __init__(self):
        # Initial checks
        monitorParams()
        cleanMemory()

        # Get parameters
        print('Loading configuration parameters ...\n')
        params = rospy.get_param('~params')
        self.classes = params['output']['classes']
        self.conf = params['model_params']['conf']
        self.overlap = params['model_params']['overlap']
        modelName = params['model_params']['model_name']
        modelPath = params['model_params']['model_path']
        modelConfig = params['model_params']['model_config']
        self.imageSize = params['image_params']['image_size']
        rawImageTopic = params['ros_topics']['raw_image_topic']
        segImageTopic = params['ros_topics']['segmented_image_topic']
        segImageVisTopic = params['ros_topics']['segmented_image_vis']

        self._queue = queue.Queue(maxsize=1)
        self._started = False
        self._should_shutdown = False
        self.visualize = rospy.get_param('~visualize')

        # Initial the segmentation module
        self.model, self.cfg = yosoInit(
            modelName, modelPath, modelConfig, self.conf, self.overlap)

        # Subscribers (to vS-Graphs)
        rospy.Subscriber(rawImageTopic, Image, self.add_message, queue_size=1)

        # Publishers (for vS-Graphs)
        self.publisherSeg = rospy.Publisher(
            segImageTopic, SegmenterDataMsg, queue_size=1)
        self.publisherSegVis = rospy.Publisher(
            segImageVisTopic, Image, queue_size=1)

        # ROS Bridge
        self.bridge = CvBridge()
        rospy.on_shutdown(self.stop)
        self.start()
    
    def start(self):
        """Start worker processing queue."""
        if not self._started:
            self._started = True
            self._thread = threading.Thread(target=self.segmentation)
            self._thread.start()
    
    def stop(self):
        """Stop worker from processing queue."""
        if self._started:
            self._should_shutdown = True
            self._thread.join()

        self._started = False
        self._should_shutdown = False

    def add_message(self, msg):
        """Add new message to queue."""
        if not self._queue.full():
            self._queue.put(msg, block=False, timeout=False)
    
    def spin(self):
        """Wait for ros to shutdown or worker to exit."""
        if not self._started:
            return

        while self._thread.is_alive() and not self._should_shutdown:
            time.sleep(1.0e-2)

        self.stop()

    def segmentation(self):
        while not self._should_shutdown:
            try:
                imageMessage = self._queue.get(timeout=0.1)
                # Parse the input data
                # keyFrameId = imageMessage.keyFrameId
                # keyFrameImage = imageMessage.keyFrameImage

                # Convert the ROS Image message to a CV2 image
                cvImage = self.bridge.imgmsg_to_cv2(imageMessage, "bgr8")

                # Processing
                filteredSegments, filteredProbs = yosoSegmenter(
                    cvImage, self.model, self.classes)
                if self.visualize:
                    segmentedImage = yosoVisualizer(cvImage, filteredSegments, self.cfg)
                segmentedUncImage = entropyVisualizer(filteredSegments["sem_seg"])

                # Convert to ROS message
                pcdProbabilities = probabilities2ROSMsg(filteredProbs,
                                                        imageMessage.header.stamp,
                                                        imageMessage.header.frame_id)

                # Create a header with the current time
                header = Header()
                header.stamp = imageMessage.header.stamp

                # Publish the processed image to vS-Graphs
                segmenterData = SegmenterDataMsg()
                segmenterData.header = header
                # segmenterData.keyFrameId = keyFrameId
                if self.visualize:
                    segmenterData.segmentedImage = self.bridge.cv2_to_imgmsg(
                        segmentedImage, "rgb8")
                segmenterData.segmentedImageUncertainty = self.bridge.cv2_to_imgmsg(
                    segmentedUncImage, "bgr8")
                segmenterData.segmentedImageProbability = pcdProbabilities
                self.publisherSeg.publish(segmenterData)
                
                if self.visualize:
                # Publish the processed image for visualization
                    visualizationImgMsg = Image()
                    visualizationImgMsg.header = header
                    visualizationImgMsg = segmenterData.segmentedImage
                    self.publisherSegVis.publish(visualizationImgMsg)

            except queue.Empty:
                    continue
            
            except CvBridgeError as e:
                rospy.logerr("CvBridge Error: {0}".format(e))


# Run the program
if __name__ == '__main__':
    # Initialization
    rospy.init_node('segmenter', anonymous=False)
    segmenter = Segmenter()
    rospy.spin()
