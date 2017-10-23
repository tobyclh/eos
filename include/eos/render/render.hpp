/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/render/render.hpp
 *
 * Copyright 2014, 2015 Patrik Huber
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
#pragma once

#ifndef RENDER_HPP_
#define RENDER_HPP_

#include "eos/core/Mesh.hpp"

#include "eos/render/detail/render_detail.hpp"
#include "eos/render/utils.hpp"
#include "eos/render/shader.hpp"
#include "eos/fitting/RenderingParameters.hpp"

#include "opencv2/core/core.hpp"

#include "boost/optional.hpp"

#include <array>
#include <vector>
#include <memory>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
namespace eos {
	namespace render {

/**
 * This file implements a software renderer conforming to OpenGL conventions. The
 * following are implementation notes, mostly for reference, or as a reminder of
 * what exactly is going on. Don't try to understand them :-)
 *
 * The renderer was initially based on code by Wojciech Sterna
 * (http://maxest.gct-game.net/content/vainmoinen/index.html), however, it has since
 * then been completely rewritten. Still I'd like to thank him for making his code
 * available and bravely answering my questions via email.
 *
 * Coordinate systems:
 * When specifying the vertices: +x = right, +y = up, we look into -z.
 * So z = 0.5 is in front of 0.0.
 * Z-Buffer:
 *
 * Shirley: Specify n and f with negative values. which makes sense b/c the points
 * are along the -z axis.
 * Consequences: notably: orthogonal(2, 3): Shirley has denominator (n-f).
 * In what space are the points in Shirley after this?
 * OGL: We're in the orthographic viewing volume looking down -z.
 * However, n and f are specified positive.

 * B/c the 3D points in front of the cam obviously still have negative z values, the
 * z-value is negated. So: n = 0.1, f = 100; With the given OpenGL ortho matrix,
 * it means a point on the near-plane which will have z = -0.1 will land up
 * on z_clip (which equals z_ndc with ortho because w=1) = -1, and a point on
 * the far plane z = -100 will have z_ndc = +1.
 *
 * That's also why in the perspective case, w_clip is set to -z_eye because
 * to project a point the formula is $x_p = (-n * x_e)/z_e$ (because our near is
 * specified with positive values, but the near-plane is _really_ at -n); but now we
 * just move the minus-sign to the denominator, $x_p = (n * x_e)/-z_e$, so in the projection matrix we can use
 * the (positive) n and f values and afterwards we divide by w = -z_e.
 *
 * http://www.songho.ca/opengl/gl_projectionmatrix.html
 *
 * Random notes:
 * clip-space: after applying the projection matrix.
 * ndc: after division by w
 * NDC cube: the range of x-coordinate from [l, r] to [-1, 1], the y-coordinate from [b, t] to [-1, 1] and the z-coordinate from [n, f] to [-1, 1].
 *
 * Note/Todo: I read that in screen space, OpenGL transform the z-values again to be between 0 and 1?
 *
 * In contrast to OGL, this renderer doesn't have state, it's just a function that gets called with all
 * necessary parameters. It's easiest for our purposes.
 *
 * Here's the whole rendering pipeline:
 * Model space
 * -> model transforms
 * World space
 * -> camera (view/eye) transform
 * View / eye / camera space ("truncated pyramid frustum". In case of ortho, it's already rectangular.)
 * -> perspective/ortho projection
 * Clip coords (x_c, y_c, z_c, w_c); the z-axis is flipped now. z [z=-n, z=-f] is mapped to [-1, +1] in case of ortho, but not yet in case of persp (it's also flipped though), but the not-[-1,1]-range is fine as we test against w_c. I.e. the larger the z-value, the further back we are.
 * Do frustum culling (clipping) here. Test the clip-coords with w_c, and discard if a tri is completely outside.
 * Of the partially visible tris, clip them against the near-plane and construct the visible part of the triangle.
 * We only do this for the near-plane here. Clipping to the near plane must be done here because after w-division triangles crossing it would get distorted.
 * "Then, OpenGL will reconstruct the edges of the polygon where clipping occurs."
 * -> Then divide by the w component of the clip coordinates
 * NDC. (now only 3D vectors: [x_ndc, y_ndc, z_ndc]). nearest points have z=-1, points on far plane have z=+1.
 * -> window transform. (also, OGL does some more to the z-buffer?)
 * Screen / window space
 * Directly after window-transform (still processing triangles), do backface culling with areVerticesCCWInScreenSpace()
 * Directly afterwards we calculate the triangle's bounding box and clip x/y (screen) against 0 and the viewport width/height.
 * Rasterising: Clipping against the far plane here by only drawing those pixels with a z-value of <= 1.0f.
 *
 * OGL: "both clipping (frustum culling) and NDC transformations are integrated into GL_PROJECTION matrix"
 *
 * Note: In both the ortho and persp case, points at z=-n end up at -1, z=-f at +1. In case of persp proj., this happens only after the divide by w.
 */

/**
 * Renders the given mesh onto a 2D image using 4x4 model-view and
 * projection matrices. Conforms to OpenGL conventions.
 *
 * @param[in] mesh A 3D mesh.
 * @param[in] model_view_matrix A 4x4 OpenGL model-view matrix.
 * @param[in] projection_matrix A 4x4 orthographic or perspective OpenGL projection matrix.
 * @param[in] viewport_width Screen width.
 * @param[in] viewport_height Screen height.
 * @param[in] texture An optional texture map. If not given, vertex-colouring is used.
 * @param[in] enable_backface_culling Whether the renderer should perform backface culling. If true, only draw triangles with vertices ordered CCW in screen-space.
 * @param[in] enable_near_clipping Whether vertices should be clipped against the near plane.
 * @param[in] enable_far_clipping Whether vertices should be clipped against the far plane.
 * @return A pair with the colourbuffer as its first element and the depthbuffer as the second element.
 */
inline std::pair<cv::Mat, cv::Mat> render(const core::Mesh& mesh, glm::tmat4x4<float> model_view_matrix, glm::tmat4x4<float> projection_matrix, int viewport_width, int viewport_height, const boost::optional<Texture>& texture, bool enable_backface_culling = false, bool enable_near_clipping = true, bool enable_far_clipping = true)
{
	// Some internal documentation / old todos or notes:
	// maybe change and pass depthBuffer as an optional arg (&?), because usually we never need it outside the renderer. Or maybe even a getDepthBuffer().
	// modelViewMatrix goes to eye-space (camera space), projection does ortho or perspective proj.
	// bool enable_texturing = false; Maybe re-add later, not sure
	// take a cv::Mat texture instead and convert to Texture internally? no, we don't want to recreate mipmap levels on each render() call.

	assert(mesh.vertices.size() == mesh.colors.size() || mesh.colors.empty()); // The number of vertices has to be equal for both shape and colour, or, alternatively, it has to be a shape-only model.
	assert(mesh.vertices.size() == mesh.texcoords.size() || mesh.texcoords.empty()); // same for the texcoords
	// another assert: If cv::Mat texture != empty, then we need texcoords?

	using cv::Mat;
	using std::vector;

	Mat colourbuffer = Mat::zeros(viewport_height, viewport_width, CV_8UC4); // make sure it's CV_8UC4?
	Mat depthbuffer = std::numeric_limits<float>::max() * Mat::ones(viewport_height, viewport_width, CV_64FC1);

	// Vertex shader:
	//processedVertex = shade(Vertex); // processedVertex : pos, col, tex, texweight
	// Assemble the vertices, project to clip space, and store as detail::Vertex (the internal representation):
	vector<detail::Vertex<float>> clipspace_vertices;
	clipspace_vertices.reserve(mesh.vertices.size());
	for (int i = 0; i < mesh.vertices.size(); ++i) { // "previously": mesh.vertex
		glm::tvec4<float> clipspace_coords = projection_matrix * model_view_matrix * mesh.vertices[i];
		glm::tvec3<float> vertex_colour;
		if (mesh.colors.empty()) {
			vertex_colour = glm::tvec3<float>(0.5f, 0.5f, 0.5f);
		}
		else {
			vertex_colour = mesh.colors[i];
		}
		clipspace_vertices.push_back(detail::Vertex<float>{clipspace_coords, vertex_colour, mesh.texcoords[i]});
	}

	// All vertices are in clip-space now.
	// Prepare the rasterisation stage.
	// For every vertex/tri:
	vector<detail::TriangleToRasterize> triangles_to_raster;
	for (const auto& tri_indices : mesh.tvi) {
		// Todo: Split this whole stuff up. Make a "clip" function, ... rename "processProspective..".. what is "process"... get rid of "continue;"-stuff by moving stuff inside process...
		// classify vertices visibility with respect to the planes of the view frustum
		// we're in clip-coords (NDC), so just check if outside [-1, 1] x ...
		// Actually we're in clip-coords and it's not the same as NDC. We're only in NDC after the division by w.
		// We should do the clipping in clip-coords though. See http://www.songho.ca/opengl/gl_projectionmatrix.html for more details.
		// However, when comparing against w_c below, we might run into the trouble of the sign again in the affine case.
		// 'w' is always positive, as it is -z_camspace, and all z_camspace are negative.
		unsigned char visibility_bits[3];
		for (unsigned char k = 0; k < 3; k++)
		{
			visibility_bits[k] = 0;
			float x_cc = clipspace_vertices[tri_indices[k]].position[0];
			float y_cc = clipspace_vertices[tri_indices[k]].position[1];
			float z_cc = clipspace_vertices[tri_indices[k]].position[2];
			float w_cc = clipspace_vertices[tri_indices[k]].position[3];
			if (x_cc < -w_cc)			// true if outside of view frustum. False if on or inside the plane.
				visibility_bits[k] |= 1;	// set bit if outside of frustum
			if (x_cc > w_cc)
				visibility_bits[k] |= 2;
			if (y_cc < -w_cc)
				visibility_bits[k] |= 4;
			if (y_cc > w_cc)
				visibility_bits[k] |= 8;
			if (enable_near_clipping && z_cc < -w_cc) // near plane frustum clipping
				visibility_bits[k] |= 16;
			if (enable_far_clipping && z_cc > w_cc) // far plane frustum clipping
				visibility_bits[k] |= 32;
		} // if all bits are 0, then it's inside the frustum
		// all vertices are not visible - reject the triangle.
		if ((visibility_bits[0] & visibility_bits[1] & visibility_bits[2]) > 0)
		{
			continue;
		}
		// all vertices are visible - pass the whole triangle to the rasterizer. = All bits of all 3 triangles are 0.
		if ((visibility_bits[0] | visibility_bits[1] | visibility_bits[2]) == 0)
		{
			boost::optional<detail::TriangleToRasterize> t = detail::process_prospective_tri(clipspace_vertices[tri_indices[0]], clipspace_vertices[tri_indices[1]], clipspace_vertices[tri_indices[2]], viewport_width, viewport_height, enable_backface_culling);
			if (t) {
				triangles_to_raster.push_back(*t);
			}
			continue;
		}
		// at this moment the triangle is known to be intersecting one of the view frustum's planes
		std::vector<detail::Vertex<float>> vertices;
		vertices.push_back(clipspace_vertices[tri_indices[0]]);
		vertices.push_back(clipspace_vertices[tri_indices[1]]);
		vertices.push_back(clipspace_vertices[tri_indices[2]]);
		// split the triangle if it intersects the near plane:
		if (enable_near_clipping)
		{
			vertices = detail::clip_polygon_to_plane_in_4d(vertices, glm::tvec4<float>(0.0f, 0.0f, -1.0f, -1.0f)); // "Normal" (or "4D hyperplane") of the near-plane. I tested it and it works like this but I'm a little bit unsure because Songho says the normal of the near-plane is (0,0,-1,1) (maybe I have to switch around the < 0 checks in the function?)
		}

		// triangulation of the polygon formed of vertices array
		if (vertices.size() >= 3)
		{
			for (unsigned char k = 0; k < vertices.size() - 2; k++)
			{
				boost::optional<detail::TriangleToRasterize> t = detail::process_prospective_tri(vertices[0], vertices[1 + k], vertices[2 + k], viewport_width, viewport_height, enable_backface_culling);
				if (t) {
					triangles_to_raster.push_back(*t);
				}
			}
		}
	}

	// Fragment/pixel shader: Colour the pixel values
	for (const auto& tri : triangles_to_raster) {
		detail::raster_triangle(tri, colourbuffer, depthbuffer, texture, enable_far_clipping);
	}
	return std::make_pair(colourbuffer, depthbuffer);
};

// Function turn a cv::Mat into a texture, and return the texture ID as a GLuint for use
inline GLuint matToTexture(cv::Mat &mat, GLenum minFilter, GLenum magFilter, GLenum wrapFilter)
{
	if(!mat.isContinuous())
	{
		mat = mat.clone();
		std::cout << "Cloning new image" << std::endl;
	}
	glPixelStorei(GL_UNPACK_ALIGNMENT,1);	
	
	// Generate a number for our textureID's unique handle
	GLuint textureID;
	glGenTextures(1, &textureID);
 
	// Bind to our texture handle
	glBindTexture(GL_TEXTURE_2D, textureID);
 
	// Catch silly-mistake texture interpolation method for magnification
	if (magFilter == GL_LINEAR_MIPMAP_LINEAR  ||
	    magFilter == GL_LINEAR_MIPMAP_NEAREST ||
	    magFilter == GL_NEAREST_MIPMAP_LINEAR ||
	    magFilter == GL_NEAREST_MIPMAP_NEAREST)
	{
		cout << "You can't use MIPMAPs for magnification - setting filter to GL_LINEAR" << endl;
		magFilter = GL_LINEAR;
	}
 
	// Set texture interpolation methods for minification and magnification
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magFilter);
 
	// Set texture clamping method
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapFilter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapFilter);
 
	
	// Set incoming texture format to:
	// GL_BGR       for CV_CAP_OPENNI_BGR_IMAGE,
	// GL_LUMINANCE for CV_CAP_OPENNI_DISPARITY_MAP,
	// Work out other mappings as required ( there's a list in comments in main() )
	GLenum inputColourFormat = GL_BGR;
	if (mat.channels() == 1)
	{
		inputColourFormat = GL_LUMINANCE;
	}
	assert(inputColourFormat != GL_LUMINANCE);
 
	// Create the texture
	glTexImage2D(GL_TEXTURE_2D,     // Type of texture
	             0,                 // Pyramid level (for mip-mapping) - 0 is the top level
	             GL_RGB,            // Internal colour format to convert to
	             mat.cols,          // Image width  i.e. 640 for Kinect in standard mode
	             mat.rows,          // Image height i.e. 480 for Kinect in standard mode
	             0,                 // Border width in pixels (can either be 1 or 0)
	             inputColourFormat, // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
	             GL_UNSIGNED_BYTE,  // Image data type
	             mat.ptr());        // The actual image data itself
 
	// If we're using mipmaps then generate them. Note: This requires OpenGL 3.0 or higher
	if (minFilter == GL_LINEAR_MIPMAP_LINEAR  ||
	    minFilter == GL_LINEAR_MIPMAP_NEAREST ||
	    minFilter == GL_NEAREST_MIPMAP_LINEAR ||
	    minFilter == GL_NEAREST_MIPMAP_NEAREST)
	{
		glGenerateMipmap(GL_TEXTURE_2D);
	}
 
	return textureID;
}

struct glmv4Comparasion
{
   bool operator() (const glm::vec4& lhs, const glm::vec4& rhs) const
   {
	return 	lhs[0] < rhs[0] ||
			lhs[0] == rhs[0] && (lhs[1] < rhs[1] || lhs[1] == rhs[1] && lhs[2] < rhs[2]);
   }
};


inline cv::Mat render_gl(const core::Mesh& mesh, const fitting::RenderingParameters& rendering_params, int viewport_width, int viewport_height, cv::Mat& isomap, bool enable_backface_culling = false,  bool enable_near_clipping = true, bool enable_far_clipping = true)
{
	cv::Mat img(viewport_height, viewport_width, CV_8UC3);
	if(isomap.type() == CV_8UC4)
	{
		cvtColor(isomap, isomap, CV_BGRA2BGR);
	} 
	assert(mesh.vertices.size() == mesh.colors.size() || mesh.colors.empty()); // The number of vertices has to be equal for both shape and colour, or, alternatively, it has to be a shape-only model.
	assert(mesh.vertices.size() == mesh.texcoords.size() || mesh.texcoords.empty()); // same for the texcoords
	// another assert: If cv::Mat texture != empty, then we need texcoords?
	// assert(!mesh.colors.empty() || texture != boost::none);
	// std::cout << "isomap type " << isomap.type() << std::endl;
	glm::tmat4x4<float> model_view_matrix = rendering_params.get_modelview();
	glm::tmat4x4<float> projection_matrix = rendering_params.get_projection();
	projection_matrix[2][2] = projection_matrix[2][2]/viewport_height/viewport_width;

	// std::cout << "projection_matrix[2][2]" << projection_matrix[2][2]  << std::endl;
	if( !glfwInit() )
	{
		fprintf( stderr, "Failed to initialize GLFW\n" );
		getchar();
		return img;
	}
	
	GLFWwindow* window;
	glfwWindowHint(GLFW_SAMPLES, 4);
	// glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	// glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	// glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
	glfwWindowHint(GLFW_OPENGL_ANY_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	window = glfwCreateWindow(viewport_width, viewport_height, "EOS Testing", NULL, NULL);
	if( window == NULL ){
		fprintf( stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n" );
		getchar();
		glfwTerminate();
		return img;
	}
	glfwMakeContextCurrent(window);
	glShadeModel(GL_SMOOTH);
	// Initialize GLEW
	glewExperimental = true; // Needed for core profile
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		getchar();
		glfwTerminate();
		return img;
	}



	// Ensure we can capture the escape key being pressed below
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
	
	// Dark blue background
	glClearColor(0.0f, 0.0f, 0.2f, 0.0f);

	// // Enable depth test
	glEnable(GL_DEPTH_TEST);
	// Accept fragment if it closer to the camera than the former one
	glDepthFunc(GL_LESS); 

	GLuint VertexArrayID;
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);
	// Create and compile our GLSL program from the shaders
	GLuint programID = LoadShaders( "eos_vertex.vert", "eos_fragment.frag" );
	// Get a handle for our "MVP" uniform

	GLuint MatrixID = glGetUniformLocation(programID, "MVP");

	glm::tmat4x4<float> MVP = projection_matrix * model_view_matrix;
	// std::cout << "MVP : " << MVP[0][0] <<  std::endl;
	// std::cout << "vertex[0] : " << MVP * mesh.vertices[0] << std::endl;
	//bind the isomap with opengl 

	GLuint TextureID  = glGetUniformLocation(programID, "myTextureSampler");
	GLuint tex = matToTexture(isomap, GL_NEAREST, GL_NEAREST, GL_CLAMP);	
	glBindTexture(GL_TEXTURE_2D, tex);
	
	std::vector<unsigned int> indices;
	for(int i = 0; i < mesh.tvi.size(); i++) 
	{ 
	  for(int j = 0; j < 3; j++) 
	  { 
		unsigned int vertexIndex = (unsigned int)(mesh.tvi.at(i)[j]); 
		indices.push_back(vertexIndex); 
	  } 
	}   

	GLuint vertexbuffer;
	glGenBuffers(1, &vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec4)*mesh.vertices.size(), &mesh.vertices[0], GL_STATIC_DRAW);
	// std::cout << "sizeof(glm::vec4<float>) : " << sizeof(glm::vec4)<< std::endl;
	
	// GLuint colourbuffer;
	// glGenBuffers(1, &colourbuffer);
	// glBindBuffer(GL_ARRAY_BUFFER, colourbuffer);
	// glBufferData(GL_ARRAY_BUFFER, sizeof(glm::tvec3<GLfloat>)*colours.size(), &colours[0], GL_STATIC_DRAW);
	// std::cout << "sizeof(glm::tvec3<GLfloat>) : " << sizeof(glm::tvec3<GLfloat>) << std::endl;
	// std::cout << "sizeof(GLfloat) : " << sizeof(GLfloat) << std::endl;

	GLuint uvbuffer;
	glGenBuffers(1, &uvbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2)*mesh.texcoords.size(), &mesh.texcoords[0], GL_STATIC_DRAW);

	GLuint elementbuffer;
	glGenBuffers(1, &elementbuffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

	// //render target
	GLuint FramebufferName = 0;
	glGenFramebuffers(1, &FramebufferName);
	glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);

	GLuint renderedTexture;
	glGenTextures(1, &renderedTexture);

	glBindTexture(GL_TEXTURE_2D, renderedTexture);

	// // Give an empty image to OpenGL ( the last "0" )
	glTexImage2D(GL_TEXTURE_2D, 0,GL_RGB, viewport_width, viewport_height, 0,GL_RGB, GL_UNSIGNED_BYTE, 0);
	
	// // Poor filtering. Needed !
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	// // The depth buffer
	GLuint depthrenderbuffer;
	glGenRenderbuffers(1, &depthrenderbuffer);
	glBindRenderbuffer(GL_RENDERBUFFER, depthrenderbuffer);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, viewport_width, viewport_height);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthrenderbuffer);

	// // Set "renderedTexture" as our colour attachement #0
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, renderedTexture, 0);

	// // Set the list of draw buffers.
	GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
	glDrawBuffers(1, DrawBuffers); // "1" is the size of DrawBuffers
	// // Always check that our framebuffer is ok
	assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE);

	// Clear the screen
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Use our shader
	glUseProgram(programID);
	// std::cout << "glUseProgram(programID);"<< std::endl;

	// Send our transformation to the currently bound shader, 
	// in the "MVP" uniform
	glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, tex);
	// Set our "myTextureSampler" sampler to use Texture Unit 0
	glUniform1i(TextureID, 0);

	// 1rst attribute buffer : vertices
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glVertexAttribPointer(
		0,                  // attribute. No particular reason for 0, but must match the layout in the shader.
		4,                  // size
		GL_FLOAT,           // type
		GL_FALSE,           // normalized?
		0,                  // stride
		(void*)0            // array buffer offset
	);

	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
	glVertexAttribPointer(
		1,                                // attribute. No particular reason for 1, but must match the layout in the shader.
		2,                                // size
		GL_FLOAT,                         // type
		GL_FALSE,                         // normalized?
		0,                                // stride
		(void*)0                          // array buffer offset
	);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer);
	
	// Draw the triangles !
	glDrawElements(
		GL_TRIANGLES,      // mode
		indices.size(),    // count
		GL_UNSIGNED_INT,   // type
		(void*)0           // element array buffer offset
	);

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	// std::cout << "glDisableVertexAttribArray(1);"<< std::endl;
			// Render to our framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);
	glViewport(0,0,viewport_width, viewport_height); // Render on the whole framebuffer, complete from the lower left corner to the upper right
	

	// //use fast 4-byte alignment (default anyway) if possible
	glPixelStorei(GL_PACK_ALIGNMENT, (img.step & 3) ? 1 : 4);
	// //set length of one complete row in destination data (doesn't need to equal img.cols)
	glPixelStorei(GL_PACK_ROW_LENGTH, img.step/img.elemSize());

	glReadPixels(0, 0, img.cols, img.rows, GL_RGB, GL_UNSIGNED_BYTE, img.data);
	
	cv::flip(img, img, 0);

	// Swap buffers
	glfwSwapBuffers(window);
	glfwPollEvents();


	// Cleanup VBO and shader
	glDeleteBuffers(1, &vertexbuffer);
	glDeleteBuffers(1, &uvbuffer);
	glDeleteTextures(1, &tex);
	glDeleteProgram(programID);
	glDeleteVertexArrays(1, &VertexArrayID);
	// Close OpenGL window and terminate GLFW
	glfwTerminate();
	return img;
}


	} /* namespace render */
} /* namespace eos */

#endif /* RENDER_HPP_ */
