#include "hydra/input/input_conversion.h"

#include <glog/logging.h>

#include <opencv2/imgproc.hpp>

#include "hydra/common/global_info.h"
#include "hydra/common/semantic_color_map.h"
#include "hydra/input/input_packet.h"
#include "hydra/input/sensor.h"
#include <execution>

namespace hydra::conversions {

namespace {

inline std::string showTypeInfo(const cv::Mat& mat) {
  std::stringstream ss;
  ss << "{depth: " << mat.depth() << ", channels: " << mat.channels() << "}";
  return ss.str();
}

}  // namespace

std::unique_ptr<InputData> parseInputPacket(const InputPacket& input_packet,
                                            const bool vertices_in_world_frame) {
  if (!input_packet.sensor_input) {
    LOG(ERROR) << "[Input Conversion] Input packet has no sensor input.";
    return nullptr;
  }

  const auto& sensor_name = input_packet.sensor_input->sensor_name;
  auto sensor = GlobalInfo::instance().getSensor(sensor_name);
  if (!sensor) {
    LOG(ERROR) << "[Input Conversion] Missing sensor '" << sensor_name
               << "' for input packet @ " << input_packet.timestamp_ns << " [ns]";
    return nullptr;
  }

  auto data = std::make_unique<InputData>(sensor);
  if (!input_packet.fillInputData(*data)) {
    LOG(ERROR) << "[Input Conversion] Unable to fill input data from input packet.";
    return nullptr;
  }

  if (!normalizeData(*data)) {
    LOG(ERROR) << "[Input Conversion] Unable to normalize data.";
    return nullptr;
  }

  if (!data->getSensor().finalizeRepresentations(*data)) {
    LOG(ERROR) << "[Input Conversion] Unable to compute inputs for integration";
    return nullptr;
  }

  convertVertexMap(*data, vertices_in_world_frame);
  return data;
}

bool normalizeDepth(InputData& data) { return convertDepth(data); }

bool normalizeData(InputData& data, bool normalize_labels) {
  if (!convertDepth(data)) {
    return false;
  }

  if (!convertColor(data)) {
    return false;
  }

  // preprocessMasks(data);

  if (!convertLabels(data)) {
    return false;
  }

  postprocessLabelImage(data);

  if (!data.vertex_map.empty() && data.vertex_map.type() != CV_32FC3) {
    LOG(ERROR) << "pointcloud must be of type CV_32FC3, not "
               << showTypeInfo(data.vertex_map);
    return false;
  }

  return true;
}

// Preprocesses all detection masks to remove depth outliers.
// This function modifies 'data.features' in place.
void preprocessMasks(InputData& data)
{
    const int N = static_cast<int>(data.features.size());
    if (N == 0 || data.depth_image.empty())
        return;

    // Deterministic scheduling: one index per feature
    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);

    std::for_each(std::execution::par,
                  indices.begin(), indices.end(),
                  [&](int idx)
    {
        // ----- 1. Local copies & quick exits ---------------------------------
        const cv::Mat mask_in = data.features[idx].mask;          // CV_32F
        if (mask_in.empty()) return;

        const cv::Mat& depth = data.depth_image;                  // CV_32F

        std::vector<float> zs;
        zs.reserve(cv::countNonZero(mask_in));

        // ----- 2. Gather depth samples inside the mask -----------------------
        for (int r = 0; r < mask_in.rows; ++r)
        {
            const float* m_ptr = mask_in.ptr<float>(r);
            const float* d_ptr = depth.ptr<float>(r);

            for (int c = 0; c < mask_in.cols; ++c)
                if (m_ptr[c] != 0.0f && !std::isnan(d_ptr[c])) {
                    zs.push_back(d_ptr[c]);
                }
        }

        if (zs.size() < 4)        // too few points for a quartile estimate
            return;

        // ----- 3. Robust IQR bounds (nth_element) ----------------------------
        const size_t q1_idx = zs.size() / 4;          // 25 %
        const size_t q3_idx = (zs.size() * 3) / 4;    // 75 %

        std::nth_element(zs.begin(), zs.begin() + q1_idx, zs.end());
        const float q1 = zs[q1_idx];

        std::nth_element(zs.begin(), zs.begin() + q3_idx, zs.end());
        const float q3 = zs[q3_idx];

        const float iqr = q3 - q1;
        if (iqr < 1e-6f)          // almost flat distribution ⇒ nothing to trim
            return;

        const float lower = q1 - 2.5f * iqr;
        const float upper = q3 + 2.5f * iqr;

        // ----- 4. Build cleaned mask -----------------------------------------
        cv::Mat mask_out = mask_in.clone();           // same shape/type

        for (int r = 0; r < mask_out.rows; ++r)
        {
            float*       out_ptr = mask_out.ptr<float>(r);
            const float* d_ptr   = depth   .ptr<float>(r);

            for (int c = 0; c < mask_out.cols; ++c)
                if (out_ptr[c])                       // pixel still in mask?
                {
                    const float z = d_ptr[c];
                    if (std::isnan(z) || z < lower || z > upper)
                        out_ptr[c] = 0;               // remove outlier
                }
        }

        // ----- 5. Commit (single, thread‑safe) write -------------------------
        data.features[idx].mask = std::move(mask_out);
    });
}

// Cleans the final label image by removing depth outliers from each labeled segment.
// This function modifies 'data.label_image' in place.
void postprocessLabelImage(InputData& data) {
    if (data.label_image.empty() || data.depth_image.empty()) {
        return;
    }

    // Find the range of label IDs in the image (e.g., 1 to N)
    double min_val, max_val;
    cv::minMaxIdx(data.label_image, &min_val, &max_val);
    const int num_labels = static_cast<int>(max_val);

    const cv::Mat& depth = data.depth_image; // CV_32F
    cv::Mat& labels = data.label_image;      // CV_16U

    // We can parallelize the cleaning of each label segment
    std::vector<int> label_ids(num_labels);
    std::iota(label_ids.begin(), label_ids.end(), 1); // Fills with 1, 2, 3...

    std::for_each(std::execution::par,
                  label_ids.begin(), label_ids.end(),
                  [&](int current_label)
    {
        // 1. Gather depth samples for the current label
        std::vector<float> zs;
        for (int r = 0; r < depth.rows; ++r) {
            const ushort* l_ptr = labels.ptr<ushort>(r);
            const float* d_ptr = depth.ptr<float>(r);
            for (int c = 0; c < depth.cols; ++c) {
                if (l_ptr[c] == current_label && !std::isnan(d_ptr[c])) {
                    zs.push_back(d_ptr[c]);
                }
            }
        }

        if (zs.size() < 4) return; // Not enough points

        // 2. Robust IQR bounds (same logic as preprocessMasks)
        const size_t q1_idx = zs.size() / 4;
        const size_t q3_idx = (zs.size() * 3) / 4;

        std::nth_element(zs.begin(), zs.begin() + q1_idx, zs.end());
        const float q1 = zs[q1_idx];
        std::nth_element(zs.begin(), zs.begin() + q3_idx, zs.end());
        const float q3 = zs[q3_idx];

        const float iqr = q3 - q1;
        if (iqr < 1e-6f) return;

        const float lower = q1 - 1.5f * iqr;
        const float upper = q3 + 1.5f * iqr;

        // 3. Remove outlier pixels for this label
        // Note: This part modifies the shared 'labels' matrix.
        // Since each thread works on a different 'current_label',
        // two threads will never try to write to the same pixel,
        // but reads/writes are interleaved. This is generally safe.
        for (int r = 0; r < labels.rows; ++r) {
            ushort* l_ptr = labels.ptr<ushort>(r);
            const float* d_ptr = depth.ptr<float>(r);
            for (int c = 0; c < labels.cols; ++c) {
                if (l_ptr[c] == current_label) {
                    const float z = d_ptr[c];
                    if (std::isnan(z) || z < lower || z > upper) {
                        l_ptr[c] = 0; // Set outlier to background
                    }
                }
            }
        }
    });
}

// This function computes a non-overlapping label image from multiple masks
bool convertLabels(InputData& data) {
    using Masks = std::vector<cv::Mat>;
    const int N = data.features.size();
    if (N == 0) {
        LOG(WARNING) << "No detection features to convert to labels.";
        return false;
    }
    Masks masks_in;
    masks_in.reserve(N);
    for (const hydra::DetectionFeature& feature : data.features) {
        masks_in.emplace_back(feature.mask.clone());
    }
    const int H = masks_in[0].rows, W = masks_in[0].cols;

    // --------------------------------------------------------------------- //
    // 1) Build per-pixel overlap count  (uint8 in [0..N])
    // --------------------------------------------------------------------- //
    cv::Mat count = cv::Mat::zeros(H, W, CV_8U);
    for (const auto& m : masks_in){
      cv::Mat non_zero_mask;
      cv::compare(m, 0, non_zero_mask, cv::CMP_NE); // Create a mask where m is not 0
      cv::add(count, cv::Scalar(1), count, non_zero_mask); // Increment count by 1 where m is not 0
                                                        // Ensure count type can hold N
    }

    cv::Mat overlapped;
    cv::compare(count, 1, overlapped, cv::CMP_GT);      // 1 = exclusive, >1 = overlap

    // --------------------------------------------------------------------- //
    // 2) Representative depth for every instance (median of exclusive pixels)
    // --------------------------------------------------------------------- //
    std::vector<float> repDepth(N, std::numeric_limits<float>::infinity());

    for (int idx = 0; idx < N; ++idx)
    {
        std::vector<float> zs;
        const cv::Mat& m = masks_in[idx];

        for (int y = 0; y < H; ++y)
        {
            const float* mp = m.ptr<float>(y);
            const uchar* ov = overlapped.ptr<uchar>(y);
            const float* dp = data.depth_image.ptr<float>(y);

            for (int x = 0; x < W; ++x)
            {
                if (mp[x] != 0.0f && !ov[x])
                {
                    float z = dp[x];
                    if (!std::isnan(z)) zs.push_back(z);
                }
            }
        }
        if (zs.empty())   // the object has no exclusive pixels → use full mask
        {
            for (int y = 0; y < H; ++y)
            {
                const float* mp = m.ptr<float>(y);
                const float* dp = data.depth_image.ptr<float>(y);
                for (int x = 0; x < W; ++x)
                    if (mp[x] != 0.0f && !std::isnan(dp[x])) zs.push_back(dp[x]);
            }
        }
        if (zs.empty()) repDepth[idx] = std::numeric_limits<float>::infinity();
        else
        {
          // Find the median depth for this instance
          size_t mid = zs.size() / 2;
          std::nth_element(zs.begin(), zs.begin() + mid, zs.end());
          repDepth[idx] = zs[mid];
        }

        // if (std::isinf(repDepth[idx]))
        // {
        //   LOG(WARNING) << "No valid depth found for instance " << idx
        //                << ", using infinity as representative depth.";
        // }
    }

    // --------------------------------------------------------------------- //
    // 3) Start result as a copy, then erase “losing” pixels
    // --------------------------------------------------------------------- //
    Masks clean;
    clean.reserve(N);
    for (const auto& m : masks_in) clean.emplace_back(m.clone());

    // Helper: list of instance IDs present at (y,x)
    std::vector<int> ids_here;  ids_here.reserve(N);

    for (int y = 0; y < H; ++y)
    {
        const uchar* ov = overlapped.ptr<uchar>(y);
        float* dz = data.depth_image.ptr<float>(y);

        for (int x = 0; x < W; ++x)
        {
            // Not overlapping, skip
            if (!ov[x]) continue;

            ids_here.clear();
            for (int idx = 0; idx < N; ++idx)
                if (masks_in[idx].at<float>(y,x) != 0.0f) ids_here.push_back(idx);

            // Should not happen, but skip if only one ID is present at that pixel
            if (ids_here.size() < 2) continue;

            float z_px = dz[x];
            if (std::isnan(z_px)) z_px = std::numeric_limits<float>::quiet_NaN();   // unlikely but safe

            // Choose the ID whose representative depth is closest to z_px
            int winner = ids_here.front();
            float best = std::fabs(repDepth[winner] - z_px);
            for (size_t k = 1; k < ids_here.size(); ++k)
            {
                int id = ids_here[k];
                float diff = std::fabs(repDepth[id] - z_px);

                // little tie-break band for sensor noise
                if (diff < best)
                {
                    best   = diff;
                    winner = id;
                }
            }

            // Erase pixel from all losers
            for (int id : ids_here)
                if (id != winner) clean[id].at<float>(y,x) = 0;
        }
    }
    // ------------------------------------------------------------------
    // 4)  Stitch the non-overlapping masks into one label image
    // ------------------------------------------------------------------
    cv::Mat label_image(H, W, CV_16U, cv::Scalar(0));   // 0 = background
    for (size_t i = 0; i < clean.size(); ++i)
    {
        cv::Mat bin8;
        if (clean[i].type() != CV_8UC1) {
            cv::compare(clean[i], 0, bin8, cv::CMP_NE);
        }
        const cv::Mat& mask8 = (clean[i].type() == CV_8UC1) ? clean[i] : bin8;
        // set label (i+1) wherever this mask is 255
        label_image.setTo(static_cast<uint16_t>(i + 1), mask8);
    }
    std::swap(data.label_image, label_image);
    return true;
}


bool colorToLabels(cv::Mat& label_image, const cv::Mat& colors) {
  if (colors.empty() || colors.channels() != 3) {
    LOG(ERROR) << "color image required to decode semantic labels";
    return false;
  }

  CHECK_EQ(colors.type(), CV_8UC3);

  const auto colormap_ptr = GlobalInfo::instance().getSemanticColorMap();
  if (!colormap_ptr || !colormap_ptr->isValid()) {
    LOG(ERROR)
        << "label colormap not valid, but required for converting colors to labels!";
    return false;
  }

  cv::Mat new_label_image(colors.size(), CV_32SC1);
  for (int r = 0; r < colors.rows; ++r) {
    for (int c = 0; c < colors.cols; ++c) {
      const auto& pixel = colors.at<cv::Vec3b>(r, c);
      spark_dsg::Color color(pixel[0], pixel[1], pixel[2]);
      // this is lazy, but works out to the same invalid label we normally use
      new_label_image.at<int32_t>(r, c) =
          colormap_ptr->getLabelFromColor(color).value_or(-1);
    }
  }

  label_image = new_label_image;
  return true;
}


bool convertDepth(InputData& data) {
  if (data.depth_image.empty()) {
    return true;
  }

  if (data.depth_image.channels() != 1) {
    LOG(ERROR) << "depth image must be single-channel";
    return false;
  }

  if (data.depth_image.type() == CV_32FC1) {
    return true;  // nothing else to do
  }

  if (data.depth_image.type() != CV_16UC1) {
    LOG(ERROR) << "only CV_32FC1 or CV_16UC1 formats supported, not "
               << showTypeInfo(data.depth_image);
    return false;
  }

  cv::Mat depth_converted;
  data.depth_image.convertTo(depth_converted, CV_32FC1, 1.0e-3);
  data.depth_image = depth_converted;
  return true;
}

bool convertColor(InputData& data) {
  if (data.color_image.empty()) {
    return true;
  }

  if (data.color_image.type() != CV_8UC3) {
    LOG(ERROR) << "only 3-channel rgb images supported";
    return false;
  }
  return true;
}

void convertVertexMap(InputData& data, bool in_world_frame) {
  if (data.points_in_world_frame == in_world_frame) {
    return;
  }
  Eigen::Isometry3f transform = data.getSensorPose().cast<float>();  // world_T_sensor
  if (!in_world_frame) {
    transform = transform.inverse();  // Instead get sensor_T_world
  }
  for (int r = 0; r < data.vertex_map.rows; ++r) {
    for (int c = 0; c < data.vertex_map.cols; ++c) {
      cv::Vec3f& point = data.vertex_map.at<cv::Vec3f>(r, c);
      Eigen::Vector3f point_eigen(point[0], point[1], point[2]);
      point_eigen = transform * point_eigen;
      point[0] = point_eigen.x();
      point[1] = point_eigen.y();
      point[2] = point_eigen.z();
    }
  }
  data.points_in_world_frame = in_world_frame;
}

}  // namespace hydra::conversions
