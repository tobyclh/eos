/*
 * Eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: utils/bfm-binary-to-cereal.cpp
 *
 * Copyright 2016 Patrik Huber
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "eos/morphablemodel/MorphableModel.hpp"
#include "eos/morphablemodel/Blendshape.hpp"

#include "Eigen/Core"

#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"

#include <vector>
#include <iostream>
#include <fstream>
#include <string>

using namespace eos;
namespace po = boost::program_options;
namespace fs = boost::filesystem;
using std::cout;
using std::endl;
using std::vector;


/**
 * Reads a raw binary file created with share/convert_bfm2009_to_raw_binary.m
 * and outputs it as an eos .bin file. Optionally, an .obj file can be given -
 * the texture coordinates from that obj will then be read and used as the
 * model's texture coordinates (as the BFM comes without texture coordinates).
 */
int main(int argc, char *argv[])
{
	fs::path bfm_file, obj_file, outputfile;
	std::string file_type;
	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"display the help message")
			("input,i", po::value<fs::path>(&bfm_file)->required(),
				"input raw binary model file from Matlab script")
			("texture-coordinates,t", po::value<fs::path>(&obj_file),
				"optional .obj file to read texture coordinates from")
			("output,o", po::value<fs::path>(&outputfile)->required()->default_value("bfm2009.bin"),
				"output filename for the converted .bin file")
			;
		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
		if (vm.count("help")) {
			cout << "Usage: bfm-binary-to-cereal [options]" << endl;
			cout << desc;
			return EXIT_SUCCESS;
		}
		po::notify(vm);
	}
	catch (const po::error& e) {
		cout << "Error while parsing command-line arguments: " << e.what() << endl;
		cout << "Use --help to display a list of options." << endl;
		return EXIT_FAILURE;
	}

	std::ifstream file(bfm_file.string(), std::ios::binary);
	if (!file.is_open()) {
		std::cout << "Unable to open model file: " << bfm_file.string() << std::endl;
		return EXIT_FAILURE;
	}


	// Read the shape model - first some dimensions:
	int num_vertices = 0;
	{
		int num_vertices_times_three = 0; // the data dimension
		file.read(reinterpret_cast<char*>(&num_vertices_times_three), 4); // 1 char = 1 byte. uint32=4bytes. float64=8bytes.
		if (num_vertices_times_three % 3 != 0)
		{
			std::cout << "Shape: num_vertices_times_three % 3 != 0" << std::endl;
			return EXIT_FAILURE;
		}
		num_vertices = num_vertices_times_three / 3;
	}
	int num_shape_basis_vectors = 0;
	file.read(reinterpret_cast<char*>(&num_shape_basis_vectors), 4);

	using Eigen::VectorXf;
	using Eigen::MatrixXf;

	// Read the mean:
	VectorXf mean_shape(num_vertices * 3);
	for (int i = 0; i < num_vertices * 3; ++i) {
		float value = 0.0f;
		file.read(reinterpret_cast<char*>(&value), 4);
		mean_shape(i) = value;
	}

	// Read the orthonormal shape basis matrix:
	MatrixXf orthonormal_pca_basis_shape(num_vertices * 3, num_shape_basis_vectors); // m x n (rows x cols) = numShapeDims x numShapePcaCoeffs
	std::cout << "Loading expression basis matrix with " << orthonormal_pca_basis_shape.rows() << " rows and " << orthonormal_pca_basis_shape.cols() << " cols." << std::endl;
	for (int col = 0; col < num_shape_basis_vectors; ++col) {
		for (int row = 0; row < num_vertices * 3; ++row) {
			float value = 0.0f;
			file.read(reinterpret_cast<char*>(&value), 4);
			orthonormal_pca_basis_shape(row, col) = value;
		}
    }
    
    
	// Read the shape eigenvalues:
	VectorXf eigenvalues_shape(num_shape_basis_vectors);
	for (int i = 0; i < num_shape_basis_vectors; ++i) {
        float value = 0.0f;
		file.read(reinterpret_cast<char*>(&value), 4);
		eigenvalues_shape(i) = value;
	}
    
    file.close();

    
	//Mat normalised_expression_shape = morphablemodel::normalise_pca_basis(unnormalised_expression_shape, eigenvalues_shape);

	std::vector<morphablemodel::Blendshape> blendshapes;
	blendshapes.reserve(orthonormal_pca_basis_shape.cols());
	for (int i = 0; i < orthonormal_pca_basis_shape.cols(); ++i) {
		morphablemodel::Blendshape blendshape;
		blendshape.deformation = orthonormal_pca_basis_shape.col(i);;
		blendshape.name = std::to_string(eigenvalues_shape(i));
		blendshapes.push_back(blendshape);
	}

	std::ofstream result(outputfile.string(), std::ios::binary);
	cereal::BinaryOutputArchive output_archive(result);
    output_archive(blendshapes);
    
    cout << "Eigen: " << eigenvalues_shape << endl;
	cout << "Saved expression .bin model as " << outputfile.string() << "." << endl;
	return EXIT_SUCCESS;
}
