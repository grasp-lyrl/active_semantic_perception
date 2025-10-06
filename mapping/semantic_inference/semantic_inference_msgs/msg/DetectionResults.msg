# DetectionResult.msg
# Contains all detected objects from a single inference, along with a header and the mask.

std_msgs/Header header # Timestamp and frame_id

# Array of detected objects
semantic_inference_msgs/DetectedObject[] detections