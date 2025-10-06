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

#ifndef GEOMETRIC_TOOLS_H
#define GEOMETRIC_TOOLS_H

#include <opencv2/core/core.hpp>
#include <sophus/se3.hpp>
#include <Eigen/Core>

namespace ORB_SLAM3
{

    class KeyFrame;

    class GeometricTools
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        // Compute the Fundamental matrix between KF1 and KF2
        static Eigen::Matrix3f ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2);

        // Triangulate point with KF1 and KF2
        static bool Triangulate(Eigen::Vector3f &x_c1, Eigen::Vector3f &x_c2, Eigen::Matrix<float, 3, 4> &Tc1w, Eigen::Matrix<float, 3, 4> &Tc2w, Eigen::Vector3f &x3D);

        template <int rows, int cols>
        static bool CheckMatrices(const cv::Mat &cvMat, const Eigen::Matrix<float, rows, cols> &eigMat)
        {
            const float epsilon = 1e-3;
            // std::cout << cvMat.cols - cols << cvMat.rows - rows << std::endl;
            if (rows != cvMat.rows || cols != cvMat.cols)
            {
                std::cout << "wrong cvmat size\n";
                return false;
            }
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    if ((cvMat.at<float>(i, j) > (eigMat(i, j) + epsilon)) ||
                        (cvMat.at<float>(i, j) < (eigMat(i, j) - epsilon)))
                    {
                        std::cout << "cv mat:\n"
                                  << cvMat << std::endl;
                        std::cout << "eig mat:\n"
                                  << eigMat << std::endl;
                        return false;
                    }
            return true;
        }

        template <typename T, int rows, int cols>
        static bool CheckMatrices(const Eigen::Matrix<T, rows, cols> &eigMat1, const Eigen::Matrix<T, rows, cols> &eigMat2)
        {
            const float epsilon = 1e-3;
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    if ((eigMat1(i, j) > (eigMat2(i, j) + epsilon)) ||
                        (eigMat1(i, j) < (eigMat2(i, j) - epsilon)))
                    {
                        std::cout << "eig mat 1:\n"
                                  << eigMat1 << std::endl;
                        std::cout << "eig mat 2:\n"
                                  << eigMat2 << std::endl;
                        return false;
                    }
            return true;
        }
    };

} // namespace ORB_SLAM

#endif // GEOMETRIC_TOOLS_H
