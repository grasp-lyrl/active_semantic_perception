/**
 * This file is a modified version of a file from ORB-SLAM3.
 * 
 * Modifications Copyright (C) 2023-2025 SnT, University of Luxembourg
 * Ali Tourani, Saad Ejaz, Hriday Bavle, Jose Luis Sanchez-Lopez, and Holger Voos
 * 
 * Original Copyright (C) 2014-2021 University of Zaragoza:
 * Raúl Mur-Artal, Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez,
 * José M.M. Montiel, and Juan D. Tardós.
 * 
 * This file is part of vS-Graphs, which is free software: you can redistribute it
 * and/or modify it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 *
 * vS-Graphs is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with this program.
 * If not, see <https://www.gnu.org/licenses/>.
*/

#include "common.h"
#include <semantic_inference_msgs/DetectionResults.h>

using namespace std;

class ImageGrabber
{
public:
    ImageGrabber(const std::string world_frame, const std::string cam_frame) : tf_listener_(tf_buffer_), world_frame_(world_frame), cam_frame_(cam_frame) {}

    void GrabArUcoMarker(const aruco_msgs::MarkerArray &msg);
    void GrabSegmentation(const segmenter_ros::SegmenterDataMsg &msgSegImage);
    void GrabVoxbloxSkeletonGraph(const visualization_msgs::MarkerArray &msgSkeletonGraph);
    void GrabGNNRoomCandidates(const vs_graphs::VSGraphsAllDetectdetRooms &msgGNNRooms);
    void RecordMsg(const sensor_msgs::ImageConstPtr &msgRGB, const sensor_msgs::ImageConstPtr &msgD,
                    const sensor_msgs::PointCloud2ConstPtr &msgPC, const segmenter_ros::SegmenterDataMsg::ConstPtr &msgSeg,
                    const semantic_inference_msgs::DetectionResults::ConstPtr &msgObj);
    void GrabRGBDSegment();

private:
    std::mutex data_mutex_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    std::string world_frame_;
    std::string cam_frame_;
    std::tuple<sensor_msgs::ImageConstPtr, sensor_msgs::ImageConstPtr, sensor_msgs::PointCloud2ConstPtr, segmenter_ros::SegmenterDataMsg::ConstPtr, semantic_inference_msgs::DetectionResults::ConstPtr> latest_data_;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "RGBD");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    if (argc > 1)
        ROS_WARN("Arguments supplied via command line are ignored.");

    std::string nodeName = ros::this_node::getName();

    ros::NodeHandle nodeHandler;
    image_transport::ImageTransport imgTransport(nodeHandler);

    std::string vocFile, settingsFile, sysParamsFile;
    nodeHandler.param<std::string>(nodeName + "/voc_file", vocFile, "file_not_set");
    nodeHandler.param<std::string>(nodeName + "/settings_file", settingsFile, "file_not_set");
    nodeHandler.param<std::string>(nodeName + "/sys_params_file", sysParamsFile, "file_not_set");

    if (vocFile == "file_not_set" || settingsFile == "file_not_set")
    {
        ROS_ERROR("[Error] 'vocabulary' and 'settings' are not provided in the launch file! Exiting...");
        ros::shutdown();
        return 1;
    }

    if (sysParamsFile == "file_not_set")
    {
        ROS_ERROR("[Error] The `YAML` file containing system parameters is not provided in the launch file! Exiting...");
        ros::shutdown();
        return 1;
    }

    bool enablePangolin;
    nodeHandler.param<bool>(nodeName + "/enable_pangolin", enablePangolin, true);

    nodeHandler.param<double>(nodeName + "/yaw", yaw, 0.0);
    nodeHandler.param<double>(nodeName + "/roll", roll, 0.0);
    nodeHandler.param<double>(nodeName + "/pitch", pitch, 0.0);

    nodeHandler.param<std::string>(nodeName + "/frame_map", frameMap, "map");
    nodeHandler.param<bool>(nodeName + "/colored_pointcloud", colorPointcloud, true);
    nodeHandler.param<bool>(nodeName + "/publish_pointclouds", pubPointClouds, true);
    nodeHandler.param<std::string>(nodeName + "/cam_frame_id", cam_frame_id, "camera");
    nodeHandler.param<std::string>(nodeName + "/world_frame_id", world_frame_id, "world");
    nodeHandler.param<std::string>(nodeName + "/frame_building_component", frameBC, "plane");
    nodeHandler.param<std::string>(nodeName + "/frame_structural_element", frameSE, "room");
    nodeHandler.param<bool>(nodeName + "/static_transform", pubStaticTransform, false);

    // Initializing system threads and getting ready to process frames
    ImageGrabber igb(world_frame_id, cam_frame_id);
    sensorType = ORB_SLAM3::System::RGBD;

    pSLAM = new ORB_SLAM3::System(vocFile, settingsFile, sysParamsFile, sensorType, enablePangolin);

    // Subscribe to get raw images
    message_filters::Subscriber<sensor_msgs::Image> subImgRGB(nodeHandler, "/dominic/forward/color/image_raw", 500);
    message_filters::Subscriber<sensor_msgs::Image> subImgDepth(nodeHandler, "/dominic/forward/depth/image_rect_raw", 500);
    message_filters::Subscriber<segmenter_ros::SegmenterDataMsg> subSegmentedImage(nodeHandler, "/camera/color/image_segment", 500);
    message_filters::Subscriber<semantic_inference_msgs::DetectionResults> subObjectImage(nodeHandler, "/dominic/forward/semantic/image_raw", 500);

    // Subscribe to get pointcloud from the depth sensor
    message_filters::Subscriber<sensor_msgs::PointCloud2> subPointcloud(nodeHandler, "/camera/depth/points", 500);

    // Synchronization of raw and depth images
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::PointCloud2, segmenter_ros::SegmenterDataMsg, semantic_inference_msgs::DetectionResults>
        syncPolicy;
    message_filters::Synchronizer<syncPolicy> sync(syncPolicy(50), subImgRGB, subImgDepth, subPointcloud, subSegmentedImage, subObjectImage);
    sync.registerCallback(boost::bind(&ImageGrabber::RecordMsg, &igb, _1, _2, _3, _4, _5));

    // Subscribe to the markers detected by `aruco_ros` library
    ros::Subscriber subAruco = nodeHandler.subscribe("/aruco_marker_publisher/markers", 1,
                                                     &ImageGrabber::GrabArUcoMarker, &igb);

    // Subscriber for images obtained from the Semantic Segmentater
    // ros::Subscriber subSegmentedImage = nodeHandler.subscribe("/camera/color/image_segment", 50,
    //                                                           &ImageGrabber::GrabSegmentation, &igb);

    // Subscriber to get the mesh from voxblox
    ros::Subscriber subVoxbloxSkeletonMesh = nodeHandler.subscribe("/voxblox_skeletonizer/sparse_graph", 1,
                                                                   &ImageGrabber::GrabVoxbloxSkeletonGraph, &igb);
    
    // Subscriber to get the room candidates detected by the GNN module
    ros::Subscriber subGNNRooms = nodeHandler.subscribe("/gnn_room_detector", 1,
                                                        &ImageGrabber::GrabGNNRoomCandidates, &igb);

    setupPublishers(nodeHandler, imgTransport, nodeName);
    setupServices(nodeHandler, nodeName);

    ros::AsyncSpinner spinner(1); // Use 1 thread
    spinner.start();

    ros::Rate rate(10);

    while (ros::ok())
    {
        igb.GrabRGBDSegment();
        rate.sleep();
    }

    // Stop all threads
    pSLAM->Shutdown();
    ros::shutdown();

    return 0;
}

void ImageGrabber::RecordMsg(const sensor_msgs::ImageConstPtr &msgRGB, const sensor_msgs::ImageConstPtr &msgD,
                            const sensor_msgs::PointCloud2ConstPtr &msgPC, const segmenter_ros::SegmenterDataMsg::ConstPtr &msgSeg,
                            const semantic_inference_msgs::DetectionResults::ConstPtr &msgObj)
{
    std::lock_guard<std::mutex> lock(data_mutex_);
    // Overwrite the previous data with the newest incoming message tuple
    latest_data_ = std::make_tuple(msgRGB, msgD, msgPC, msgSeg, msgObj);
}

void ImageGrabber::GrabRGBDSegment()
{
    std::tuple<sensor_msgs::ImageConstPtr, sensor_msgs::ImageConstPtr, sensor_msgs::PointCloud2ConstPtr, segmenter_ros::SegmenterDataMsg::ConstPtr, semantic_inference_msgs::DetectionResults::ConstPtr> data_copy;
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        // Create a copy of the data tuple inside the lock
        data_copy = latest_data_;
        // Clear the shared buffer so we don't process the same message twice
        latest_data_ = {};
    }
    if (!std::get<0>(data_copy) || !std::get<1>(data_copy) || !std::get<2>(data_copy) || !std::get<3>(data_copy) || !std::get<4>(data_copy))
        return;

    // Variables
    cv_bridge::CvImageConstPtr cv_ptrD, cv_ptrRGB;
    try
    {
        cv_ptrD = cv_bridge::toCvShare(std::get<1>(data_copy));
        cv_ptrRGB = cv_bridge::toCvShare(std::get<0>(data_copy));
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("[Error] Problem occured while running `cv_bridge`: %s", e.what());
        return;
    }

    // Convert pointclouds from ros to pcl format
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromROSMsg(*std::get<2>(data_copy), *cloud);

    // Tracking process sends markers found in this frame for tracking and clears the buffer
    cv_bridge::CvImageConstPtr cv_imgSeg;

    try
    {
        cv_imgSeg = cv_bridge::toCvCopy(std::get<3>(data_copy)->segmentedImageUncertainty, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    // Convert to PCL PointCloud2 from `sensor_msgs` PointCloud2
    pcl::PCLPointCloud2::Ptr pclPc2SegPrb(new pcl::PCLPointCloud2);
    pcl_conversions::toPCL(std::get<3>(data_copy)->segmentedImageProbability, *pclPc2SegPrb);

    ros::WallRate tf_wait_rate(10.0);
    bool have_transform = false;
    std::string err_str;
    size_t attempt_number = 0;
    uint8_t max_tries = 5;
    ros::Time timeLookup = std::get<0>(data_copy)->header.stamp;
    while (ros::ok()) {
        if (max_tries && attempt_number >= max_tries) {
            break;
        }

        if (tf_buffer_.canTransform(world_frame_, cam_frame_, timeLookup, ros::Duration(0), &err_str)) {
            have_transform = true;
            break;
        }

        ++attempt_number;
        tf_wait_rate.sleep();
    }
    if(!have_transform) {
        ROS_ERROR("Failed to get transform from '%s' to '%s' at time %f: %s", cam_frame_.c_str(), world_frame_.c_str(), timeLookup.toSec(), err_str.c_str());
        return;
    }
    geometry_msgs::TransformStamped transformStamped;
    transformStamped = tf_buffer_.lookupTransform(world_frame_, cam_frame_, timeLookup);
    tf2::Quaternion tf_quat;
    tf2::fromMsg(transformStamped.transform.rotation, tf_quat);
    tf2::Matrix3x3 rot_cTw(tf_quat);
    tf2::Vector3 trans_cTw(transformStamped.transform.translation.x,
                            transformStamped.transform.translation.y,
                            transformStamped.transform.translation.z);
    tf2::Transform tf_cTw(rot_cTw, trans_cTw);

    // Create the tuple to be appended to the segmentedImageBuffer
    auto tuple_ptr = std::make_shared<std::tuple<pcl::PointCloud<pcl::PointXYZRGB>::Ptr, cv::Mat, pcl::PCLPointCloud2::Ptr, tf2::Transform, semantic_inference_msgs::DetectionResults::ConstPtr>>
                        (cloud, cv_imgSeg->image, pclPc2SegPrb, tf_cTw, std::get<4>(data_copy));

    // Add the segmented image to a buffer to be processed in the SemanticSegmentation thread
    pSLAM->addSegmentedImage(tuple_ptr);
    ros::Time msgTime = cv_ptrRGB->header.stamp;
    publishTopics(msgTime);
}

/**
 * @brief Callback function to get the markers detected by the `aruco_ros` library
 *
 * @param msgMarkerArray The markers detected by the `aruco_ros` library
 */
void ImageGrabber::GrabArUcoMarker(const aruco_msgs::MarkerArray &msgMarkerArray)
{
    // Pass the visited markers to a buffer to be processed later
    addMarkersToBuffer(msgMarkerArray);
}

/**
 * @brief Callback function to get scene segmentation results from the SemanticSegmenter module
 *
 * @param msgSegImage The segmentation results from the SemanticSegmenter
 */
// void ImageGrabber::GrabSegmentation(const segmenter_ros::SegmenterDataMsg &msgSegImage)
// {
//     // Fetch the segmentation results
//     cv_bridge::CvImageConstPtr cv_imgSeg;
//     uint64_t keyFrameId = msgSegImage.keyFrameId.data;

//     try
//     {
//         cv_imgSeg = cv_bridge::toCvCopy(msgSegImage.segmentedImageUncertainty, sensor_msgs::image_encodings::BGR8);
//     }
//     catch (cv_bridge::Exception &e)
//     {
//         ROS_ERROR("cv_bridge exception: %s", e.what());
//         return;
//     }

//     // Convert to PCL PointCloud2 from `sensor_msgs` PointCloud2
//     pcl::PCLPointCloud2::Ptr pclPc2SegPrb(new pcl::PCLPointCloud2);
//     pcl_conversions::toPCL(msgSegImage.segmentedImageProbability, *pclPc2SegPrb);

//     // Create the tuple to be appended to the segmentedImageBuffer
//     std::tuple<uint64_t, cv::Mat, pcl::PCLPointCloud2::Ptr> tuple(keyFrameId, cv_imgSeg->image, pclPc2SegPrb);

//     // Add the segmented image to a buffer to be processed in the SemanticSegmentation thread
//     pSLAM->addSegmentedImage(&tuple);
// }

/**
 * @brief Callback function to get the skeleton graph from the `voxblox` module
 *
 * @param msgSkeletonGraphs The skeleton graph from the `voxblox` module
 */
void ImageGrabber::GrabVoxbloxSkeletonGraph(const visualization_msgs::MarkerArray &msgSkeletonGraphs)
{
    // Pass the skeleton graph to a buffer to be processed by the SemanticSegmentation thread
    setVoxbloxSkeletonCluster(msgSkeletonGraphs);
}

/**
 * @brief Callback function to get the room candidates detected by the GNN module
 *
 * @param msgGNNRooms The room candidates detected by the GNN module
 */
void ImageGrabber::GrabGNNRoomCandidates(const vs_graphs::VSGraphsAllDetectdetRooms &msgGNNRooms)
{
    // Set the GNN room candidates in the SLAM system
    setGNNBasedRoomCandidates(msgGNNRooms);
}