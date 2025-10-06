# DetectedObject.msg
# Represents a single detected object with its mask, bounding box, confidence, and label.

# Mask for the instance
sensor_msgs/Image mask

# Bounding Box (xyxy format: xmin, ymin, xmax, ymax)
# Assumes pixel coordinates or normalized coordinates (clarify in your usage)
float32[4] bbox_xyxy

# Confidence score for the detection
float32 confidence

# Class label for the detected object (e.g., "car", "person")
string label
