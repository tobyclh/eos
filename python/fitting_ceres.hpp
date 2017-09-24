#pragma once

#define NUM_SHAPE_COEFF 199
#define NUM_COLOR_COEFF 199
#define NUM_BLENDSHAPE_COEFF 29

#include "eos/core/Landmark.hpp"
#include "eos/core/LandmarkMapper.hpp"
#include "eos/core/Mesh.hpp"
#include "eos/fitting/RenderingParameters.hpp"
#include "eos/fitting/blendshape_fitting.hpp"
#include "eos/fitting/ceres_nonlinear.hpp"
#include "eos/fitting/closest_edge_fitting.hpp"
#include "eos/fitting/contour_correspondence.hpp"
#include "eos/fitting/linear_shape_fitting.hpp"
#include "eos/fitting/orthographic_camera_estimation_linear.hpp"
#include "eos/morphablemodel/Blendshape.hpp"
#include "eos/morphablemodel/EdgeTopology.hpp"
#include "eos/morphablemodel/MorphableModel.hpp"

#include "opencv2/core/core.hpp"

#include "ceres/autodiff_cost_function.h"
#include "ceres/cost_function.h"
#include "ceres/cubic_interpolation.h"
#include "ceres/local_parameterization.h"
#include "ceres/problem.h"
#include "ceres/rotation.h"
#include "ceres/solver.h"

#include "opencv2/core/core.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

namespace eos {
namespace fitting {

using namespace ceres;
inline std::tuple<core::Mesh, glm::tmat4x4<double>, glm::tmat4x4<double>>
fit_shape_and_pose_ceres(const morphablemodel::MorphableModel& morphable_model,
                         const std::vector<morphablemodel::Blendshape>& blendshapes,
                         const core::LandmarkCollection<cv::Vec2f>& landmarks,
                         const core::LandmarkMapper& landmark_mapper, const cv::Mat& image,
                         const fitting::ContourLandmarks& ibug_contour,
                         const fitting::ModelContour& model_contour, std::vector<double>& shape_coefficients,
                         std::vector<double>& blendshape_coefficients)
{
    constexpr bool use_perspective = false;

    // These will be the 2D image points and their corresponding 3D vertex id's used for the fitting:
    std::vector<cv::Vec2f> image_points; // the 2D landmark points
    std::vector<int> vertex_indices;     // their corresponding vertex indices

    // Sub-select all the landmarks which we have a mapping for (i.e. that are defined in the 3DMM):
    for (int i = 0; i < landmarks.size(); ++i)
    {
        auto converted_name = landmark_mapper.convert(landmarks[i].name);
        if (!converted_name)
        { // no mapping defined for the current landmark
            continue;
        }
        int vertex_idx = std::stoi(converted_name.get());
        // cv::Vec4f vertex = morphable_model.get_shape_model().get_mean_at_point(vertex_idx);
        vertex_indices.emplace_back(vertex_idx);
        image_points.emplace_back(landmarks[i].coordinates);
    }

    // Estimate the camera (pose) from the 2D - 3D point correspondences
    std::stringstream fitting_log;
    auto start = std::chrono::steady_clock::now();

    std::vector<double>
        camera_rotation;       // Quaternion, [w x y z]. Todo: Actually, use std::array for all of these.
    camera_rotation.resize(4); // initialises with zeros
    camera_rotation[0] = 1.0;
    std::vector<double> camera_translation_and_intrinsics;
    constexpr int num_cam_trans_intr_params = use_perspective ? 4 : 3;
    // Parameters for the orthographic projection: [t_x, t_y, frustum_scale]
    // And perspective projection: [t_x, t_y, t_z, fov].
    // Origin is assumed at center of image, and no lens distortions.
    // Note: Actually, we estimate the model-view matrix and not the camera position. But one defines the
    // other.
    camera_translation_and_intrinsics.resize(num_cam_trans_intr_params); // initialises with zeros
    if (use_perspective)
    {
        camera_translation_and_intrinsics[2] = -400.0;              // Move the model back (along the -z axis)
        camera_translation_and_intrinsics[3] = glm::radians(45.0f); // fov
    } else
    {
        camera_translation_and_intrinsics[2] = 110.0; // frustum_scale
    }
    std::cout << morphable_model.get_shape_model().get_num_principal_components() << " "
              << morphable_model.get_color_model().get_num_principal_components() << " " << blendshapes.size()
              << std::endl;
    shape_coefficients.resize(
        NUM_SHAPE_COEFF, 0); // Todo: Currently, the value '10' is hard-coded everywhere. Make it dynamic.
    blendshape_coefficients.resize(NUM_BLENDSHAPE_COEFF, 0);

    ceres::Problem camera_costfunction;
    for (int i = 0; i < image_points.size(); ++i)
    {
        CostFunction* cost_function = new AutoDiffCostFunction<
            fitting::LandmarkCost, 2 /* num residuals */, 4 /* camera rotation (quaternion) */,
            num_cam_trans_intr_params /* camera translation & fov/frustum_scale */,
            NUM_SHAPE_COEFF /* shape-coeffs */, NUM_BLENDSHAPE_COEFF /* bs-coeffs */>(
            new fitting::LandmarkCost(morphable_model.get_shape_model(), blendshapes, image_points[i],
                                      vertex_indices[i], image.cols, image.rows, use_perspective));
        camera_costfunction.AddResidualBlock(cost_function, NULL, &camera_rotation[0],
                                             &camera_translation_and_intrinsics[0], &shape_coefficients[0],
                                             &blendshape_coefficients[0]);
    }
    camera_costfunction.SetParameterBlockConstant(&shape_coefficients[0]); // keep the shape constant
    camera_costfunction.SetParameterBlockConstant(&blendshape_coefficients[0]);
    if (use_perspective)
    {
        camera_costfunction.SetParameterUpperBound(
            &camera_translation_and_intrinsics[0], 2,
            -std::numeric_limits<double>::epsilon()); // t_z has to be negative
        camera_costfunction.SetParameterLowerBound(&camera_translation_and_intrinsics[0], 3,
                                                   0.01); // fov in radians, must be > 0
    } else
    {
        camera_costfunction.SetParameterLowerBound(&camera_translation_and_intrinsics[0], 2,
                                                   1.0); // frustum_scale must be > 0
    }
    ceres::QuaternionParameterization* camera_fit_quaternion_parameterisation =
        new QuaternionParameterization;
    camera_costfunction.SetParameterization(&camera_rotation[0], camera_fit_quaternion_parameterisation);

    Solver::Options solver_options;
    solver_options.linear_solver_type = ITERATIVE_SCHUR;
    // solver_options.linear_solver_type = ceres::DENSE_QR;
    // solver_options.minimizer_type = ceres::TRUST_REGION; // default I think
    // solver_options.minimizer_type = ceres::LINE_SEARCH;
    // solver_options.preconditioner_type = SCHUR_JACOBI;
    solver_options.num_threads = 8;
    // solver_options.num_linear_solver_threads = 8; // only SPARSE_SCHUR can use this
    solver_options.minimizer_progress_to_stdout = true;
    // solver_options.max_num_iterations = 100;
    // solver_options.use_explicit_schur_complement = true;
    Solver::Summary solver_summary;
    Solve(solver_options, &camera_costfunction, &solver_summary);
    std::cout << solver_summary.BriefReport() << "\n";
    auto end = std::chrono::steady_clock::now();

    // Draw the mean-face landmarks projected using the estimated camera:
    // Construct the rotation & translation (model-view) matrices, projection matrix, and viewport:
    glm::dquat estimated_rotation(camera_rotation[0], camera_rotation[1], camera_rotation[2],
                                  camera_rotation[3]);
    auto rot_mtx = glm::mat4_cast(estimated_rotation);
    const double aspect = static_cast<double>(image.cols) / image.rows;
    auto get_translation_matrix = [](auto&& camera_translation_and_intrinsics, auto&& use_perspective) {
        if (use_perspective)
        {
            return glm::translate(glm::dvec3(camera_translation_and_intrinsics[0],
                                             camera_translation_and_intrinsics[1],
                                             camera_translation_and_intrinsics[2]));
        } else
        {
            return glm::translate(
                glm::dvec3(camera_translation_and_intrinsics[0], camera_translation_and_intrinsics[1], 0.0));
        }
    };
    auto get_projection_matrix = [](auto&& camera_translation_and_intrinsics, auto&& aspect,
                                    auto&& use_perspective) {
        if (use_perspective)
        {
            const auto& focal = camera_translation_and_intrinsics[3];
            return glm::perspective(focal, aspect, 0.1, 1000.0);
        } else
        {
            const auto& frustum_scale = camera_translation_and_intrinsics[2];
            return glm::ortho(-1.0 * aspect * frustum_scale, 1.0 * aspect * frustum_scale,
                              -1.0 * frustum_scale, 1.0 * frustum_scale);
        }
    };
    auto t_mtx = get_translation_matrix(camera_translation_and_intrinsics, use_perspective);
    auto projection_mtx = get_projection_matrix(camera_translation_and_intrinsics, aspect, use_perspective);
    const glm::dvec4 viewport(0, image.rows, image.cols, -image.rows); // OpenCV convention

    auto euler_angles = glm::eulerAngles(estimated_rotation); // returns [P, Y, R]
    // fitting_log << "Pose fit with mean shape:\tYaw " << glm::degrees(euler_angles[1]) << ", Pitch " <<
    // glm::degrees(euler_angles[0]) << ", Roll " << glm::degrees(euler_angles[2]) << "; t & f: " <<
    // camera_translation_and_intrinsics << '\n';
    // fitting_log << "Ceres took: " << std::chrono::duration_cast<std::chrono::milliseconds>(end -
    // start).count() << "ms.\n";

    // Contour fitting:
    // These are the additional contour-correspondences we're going to find and then use:
    std::vector<cv::Vec2f> image_points_contour; // the 2D landmark points
    std::vector<int> vertex_indices_contour;     // their corresponding 3D vertex indices
    // For each 2D contour landmark, get the corresponding 3D vertex point and vertex id:
    std::tie(image_points_contour, std::ignore, vertex_indices_contour) =
        fitting::get_contour_correspondences(landmarks, ibug_contour, model_contour,
                                             glm::degrees(euler_angles[1]), morphable_model.get_mean(),
                                             t_mtx * rot_mtx, projection_mtx, viewport);
    using eos::fitting::concat;
    vertex_indices = concat(vertex_indices, vertex_indices_contour);
    image_points = concat(image_points, image_points_contour);

    // Full fitting - Estimate shape and pose, given the previous pose estimate:
    start = std::chrono::steady_clock::now();
    ceres::Problem fitting_costfunction;
    // Landmark constraint:
    for (int i = 0; i < image_points.size(); ++i)
    {
        CostFunction* cost_function = new AutoDiffCostFunction<
            fitting::LandmarkCost, 2 /* num residuals */, 4 /* camera rotation (quaternion) */,
            num_cam_trans_intr_params /* camera translation & focal length */,
            NUM_SHAPE_COEFF /* shape-coeffs */, NUM_BLENDSHAPE_COEFF /* bs-coeffs */>(
            new fitting::LandmarkCost(morphable_model.get_shape_model(), blendshapes, image_points[i],
                                      vertex_indices[i], image.cols, image.rows, use_perspective));
        fitting_costfunction.AddResidualBlock(cost_function, NULL, &camera_rotation[0],
                                              &camera_translation_and_intrinsics[0], &shape_coefficients[0],
                                              &blendshape_coefficients[0]);
    }
    // Shape prior
    CostFunction* shape_prior_cost =
        new AutoDiffCostFunction<fitting::PriorCost, NUM_SHAPE_COEFF /* num residuals */,
                                 NUM_SHAPE_COEFF /* shape-coeffs */>(
            new fitting::PriorCost(NUM_SHAPE_COEFF, 100.0));
    fitting_costfunction.AddResidualBlock(shape_prior_cost, NULL, &shape_coefficients[0]);
    for (int i = 0; i < NUM_SHAPE_COEFF; ++i)
    {
        fitting_costfunction.SetParameterLowerBound(&shape_coefficients[0], i, -3.0);
        fitting_costfunction.SetParameterUpperBound(&shape_coefficients[0], i, 3.0);
    }
    // Prior and constraints on blendshapes:
    CostFunction* blendshapes_prior_cost =
        new AutoDiffCostFunction<fitting::PriorCost, NUM_BLENDSHAPE_COEFF /* num residuals */,
                                 NUM_BLENDSHAPE_COEFF /* bs-coeffs */>(
            new fitting::PriorCost(NUM_BLENDSHAPE_COEFF, 100.0));
    fitting_costfunction.AddResidualBlock(blendshapes_prior_cost, NULL, &blendshape_coefficients[0]);
    for (int i = 0; i < NUM_BLENDSHAPE_COEFF; ++i)
    {
        fitting_costfunction.SetParameterLowerBound(&blendshape_coefficients[0], i, 0.0);
    }
    // Some constraints on the camera translation and fov/scale:
    if (use_perspective)
    {
        fitting_costfunction.SetParameterUpperBound(
            &camera_translation_and_intrinsics[0], 2,
            -std::numeric_limits<double>::epsilon()); // t_z has to be negative
        fitting_costfunction.SetParameterLowerBound(&camera_translation_and_intrinsics[0], 3,
                                                    0.01); // fov in radians, must be > 0
    } else
    {
        fitting_costfunction.SetParameterLowerBound(&camera_translation_and_intrinsics[0], 2,
                                                    1.0); // frustum_scale must be > 0
    }

    QuaternionParameterization* full_fit_quaternion_parameterisation = new QuaternionParameterization;
    fitting_costfunction.SetParameterization(&camera_rotation[0], full_fit_quaternion_parameterisation);

    // Colour model fitting:
    std::vector<double> colour_coefficients;
    colour_coefficients.resize(NUM_COLOR_COEFF);
    // Add a residual for each vertex:
    // for (int i = 0; i < morphable_model.get_shape_model().get_data_dimension() / 3; ++i)
    //{
    //	CostFunction* cost_function = new AutoDiffCostFunction<fitting::ImageCost, 3 /* Residuals: [R, G, B]
    //*/, 4 /* camera rotation (quaternion) */, num_cam_trans_intr_params /* camera translation & focal length
    //*/, NUM_SHAPE_COEFF /* shape-coeffs */, NUM_BLENDSHAPE_COEFF /* bs-coeffs */, NUM_COLOR_COEFF /* colour
    // coeffs */>(new fitting::ImageCost(morphable_model, blendshapes, image, i, use_perspective));
    //	fitting_costfunction.AddResidualBlock(cost_function, NULL, &camera_rotation[0],
    //&camera_translation_and_intrinsics[0], &shape_coefficients[0], &blendshape_coefficients[0],
    //&colour_coefficients[0]);
    //}
    // Prior for the colour coefficients:
    // CostFunction* colour_prior_cost = new AutoDiffCostFunction<fitting::PriorCost, NUM_COLOR_COEFF /* num
    // residuals */, NUM_COLOR_COEFF /* colour-coeffs */>(new fitting::PriorCost(NUM_COLOR_COEFF, 75.0));
    // fitting_costfunction.AddResidualBlock(colour_prior_cost, NULL, &colour_coefficients[0]);
    // for (int i = 0; i < NUM_COLOR_COEFF; ++i)
    //{
    //	fitting_costfunction.SetParameterLowerBound(&colour_coefficients[0], i, -3.0);
    //	fitting_costfunction.SetParameterUpperBound(&colour_coefficients[0], i, 3.0);
    //}

    Solve(solver_options, &fitting_costfunction, &solver_summary);
    std::cout << solver_summary.BriefReport() << "\n";
    end = std::chrono::steady_clock::now();

    // Draw the landmarks projected using all estimated parameters:
    // Construct the rotation & translation (model-view) matrices, projection matrix, and viewport:
    estimated_rotation =
        glm::dquat(camera_rotation[0], camera_rotation[1], camera_rotation[2], camera_rotation[3]);
    rot_mtx = glm::mat4_cast(estimated_rotation);
    t_mtx = get_translation_matrix(camera_translation_and_intrinsics, use_perspective);
    projection_mtx = get_projection_matrix(camera_translation_and_intrinsics, aspect, use_perspective);

    auto vectord_to_vectorf = [](const std::vector<double>& vec) {
        return std::vector<float>(std::begin(vec), std::end(vec));
    };
    // auto shape_ceres = morphable_model.get_shape_model().draw_sample(shape_coefficients);
    auto blendshape_coeffs_float = vectord_to_vectorf(blendshape_coefficients);
    auto shape_ceres =
        morphable_model.get_shape_model().draw_sample(shape_coefficients) +
        to_matrix(blendshapes) *
            Eigen::Map<const Eigen::VectorXf>(blendshape_coeffs_float.data(), blendshape_coeffs_float.size());
    core::Mesh mesh = morphablemodel::sample_to_mesh(
        shape_ceres, morphable_model.get_color_model().draw_sample(colour_coefficients),
        morphable_model.get_shape_model().get_triangle_list(),
        morphable_model.get_color_model().get_triangle_list(), morphable_model.get_texture_coordinates());

    estimated_rotation =
        glm::dquat(camera_rotation[0], camera_rotation[1], camera_rotation[2], camera_rotation[3]);
    euler_angles = glm::eulerAngles(estimated_rotation); // returns [P, Y, R]
    // fitting_log << "Final fit:\t\t\tYaw " << glm::degrees(euler_angles[1]) << ", Pitch " <<
    // glm::degrees(euler_angles[0]) << ", Roll " << glm::degrees(euler_angles[2]) << "; t & f: " <<
    // camera_translation_and_intrinsics << '\n';
    // fitting_log << "Ceres took: " << std::chrono::duration_cast<std::chrono::milliseconds>(end -
    // start).count() << "ms.\n";

    std::cout << fitting_log.str();
    auto tuple = std::make_tuple(mesh, (t_mtx * rot_mtx), projection_mtx);
    return tuple;
}
}
}