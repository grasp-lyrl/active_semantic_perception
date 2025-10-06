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

#ifndef CONVERTER_H
#define CONVERTER_H

#include <opencv2/core/core.hpp>

#include <Eigen/Dense>
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

#include "Thirdparty/Sophus/sophus/geometry.hpp"
#include "Thirdparty/Sophus/sophus/sim3.hpp"

namespace ORB_SLAM3
{

    class Converter
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        static std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors);

        static g2o::SE3Quat toSE3Quat(const cv::Mat &cvT);
        static g2o::SE3Quat toSE3Quat(const Sophus::SE3f &T);
        static g2o::SE3Quat toSE3Quat(const g2o::Sim3 &gSim3);

        // TODO templetize these functions
        static cv::Mat toCvMat(const g2o::SE3Quat &SE3);
        static cv::Mat toCvMat(const g2o::Sim3 &Sim3);
        static cv::Mat toCvMat(const Eigen::Matrix<double, 4, 4> &m);
        static cv::Mat toCvMat(const Eigen::Matrix<float, 4, 4> &m);
        static cv::Mat toCvMat(const Eigen::Matrix<float, 3, 4> &m);
        static cv::Mat toCvMat(const Eigen::Matrix3d &m);
        static cv::Mat toCvMat(const Eigen::Matrix<double, 3, 1> &m);
        static cv::Mat toCvMat(const Eigen::Matrix<float, 3, 1> &m);
        static cv::Mat toCvMat(const Eigen::Matrix<float, 3, 3> &m);

        static cv::Mat toCvMat(const Eigen::MatrixXf &m);
        static cv::Mat toCvMat(const Eigen::MatrixXd &m);

        static cv::Mat toCvSE3(const Eigen::Matrix<double, 3, 3> &R, const Eigen::Matrix<double, 3, 1> &t);
        static cv::Mat tocvSkewMatrix(const cv::Mat &v);

        static Eigen::Matrix<double, 3, 1> toVector3d(const cv::Mat &cvVector);
        static Eigen::Matrix<float, 3, 1> toVector3f(const cv::Mat &cvVector);
        static Eigen::Matrix<double, 3, 1> toVector3d(const cv::Point3f &cvPoint);
        static Eigen::Matrix<double, 3, 3> toMatrix3d(const cv::Mat &cvMat3);
        static Eigen::Matrix<double, 4, 4> toMatrix4d(const cv::Mat &cvMat4);
        static Eigen::Matrix<float, 3, 3> toMatrix3f(const cv::Mat &cvMat3);
        static Eigen::Matrix<float, 4, 4> toMatrix4f(const cv::Mat &cvMat4);
        static std::vector<float> toQuaternion(const cv::Mat &M);

        static bool isRotationMatrix(const cv::Mat &R);
        static std::vector<float> toEuler(const cv::Mat &R);

        // TODO: Sophus migration, to be deleted in the future
        static Sophus::SE3<float> toSophus(const cv::Mat &T);
        static Sophus::Sim3f toSophus(const g2o::Sim3 &S);
    };

} // namespace ORB_SLAM

#endif // CONVERTER_H
