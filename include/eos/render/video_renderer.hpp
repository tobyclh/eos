/*
OpenGL renderer optimized for video rendering


*/
#pragma once

#ifndef VIEWER_HPP_
#define VIEWER_HPP_

#include "eos/core/Mesh.hpp"

#include "eos/fitting/RenderingParameters.hpp"
#include "eos/render/detail/render_detail.hpp"
#include "eos/render/render.hpp"
#include "eos/render/shader.hpp"
#include "eos/render/utils.hpp"

#include "opencv2/core/core.hpp"

#include <array>
#include <memory>
#include <thread>
#include <vector>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
namespace eos {
namespace render {

class Viewer
{
public:
    std::vector<glm::vec4> vertices;   ///< 3D vertex positions.
    std::vector<unsigned int> indices; /// vertex index
    std::vector<glm::vec2> texcoords;  ///< Texture coordinates for each vertex.

    fitting::RenderingParameters pose; // in the scenario of video stream to photo this is constant
    int viewport_width;
    int viewport_height;

    cv::Mat canvas; // the canvas we will be rendering on, constant for video -> photo

    std::thread rendering;

    // OpenGL handlers
    GLFWwindow* window;
    GLuint MatrixID;
    GLuint programID;
    GLuint TextureID;
    GLuint VertexArrayID;
    GLuint FramebufferID;
    GLuint vertexbuffer;
    GLuint uvbuffer;
    GLuint elementbuffer;
    GLuint depthrenderbuffer;
    GLuint isomapTexture;
    GLuint renderedTexture;
    GLuint depthTexture;
    GLuint backgroundTexture;
    Viewer(){};

    Viewer(std::vector<glm::vec4> _vertices, std::vector<glm::vec2> _texcoords,
           std::vector<std::array<int, 3>> tvi, fitting::RenderingParameters _pose, cv::Mat _view,
           cv::Mat _isomap)
        : vertices(_vertices), texcoords(_texcoords), canvas(_view), pose(_pose), isomap(_isomap)
    {
        assert(vertices.size() == texcoords.size());
        for (int i = 0; i < tvi.size(); i++)
        {
            for (int j = 0; j < 3; j++)
            {
                unsigned int vertexIndex = (unsigned int)(tvi.at(i)[j]);
                indices.push_back(vertexIndex);
            }
        }
        viewport_width = _view.cols;
        viewport_height = _view.rows;

        if (!glfwInit())
        {
            fprintf(stderr, "Failed to initialize GLFW\n");
            getchar();
            return;
        }
        glfwWindowHint(GLFW_SAMPLES, 4);
        // glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        // glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_ANY_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        window = glfwCreateWindow(viewport_width, viewport_height, "Viewer", NULL, NULL);
        // window = glfwCreateWindow(viewport_width, viewport_height, "EOS Testing", NULL, NULL);
        if (window == NULL)
        {
            fprintf(stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 "
                            "compatible. Try the 2.1 version of the tutorials.\n");
            getchar();
            glfwTerminate();
            return;
        }
        glfwMakeContextCurrent(window);
        glShadeModel(GL_SMOOTH);
        // Initialize GLEW
        glewExperimental = true; // Needed for core profile
        if (glewInit() != GLEW_OK)
        {
            fprintf(stderr, "Failed to initialize GLEW\n");
            getchar();
            glfwTerminate();
            return;
        }
        // Ensure we can capture the escape key being pressed below

        glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);


        // // Enable depth test
        glEnable(GL_DEPTH_TEST);
        // glEnable(GL_DEPTH_CLAMP);

        // glDepthRange(-100, 10);
        // Accept fragment if it closer to the camera than the former one
        glDepthFunc(GL_LESS);
        glShadeModel(GL_SMOOTH);

        programID = LoadShaders("eos_vertex.vert", "eos_fragment.frag");
        glGenBuffers(1, &vertexbuffer);
        glGenBuffers(1, &uvbuffer);
        glGenBuffers(1, &elementbuffer);
        glGenFramebuffers(1, &FramebufferID);
        glGenTextures(1, &renderedTexture);
        glGenRenderbuffers(1, &depthrenderbuffer);
        glGenTextures(1, &depthTexture);
        glGenTextures(1, &backgroundTexture);
        glUseProgram(programID);
        return;
    }

    std::pair<cv::Mat, cv::Mat> render()
    {
        if (isomap.type() == CV_8UC4)
        {
            cvtColor(isomap, isomap, CV_BGRA2BGR);
        }
        cv::Mat img(viewport_height, viewport_width, CV_8UC3);
        cv::Mat depth(viewport_height, viewport_width, CV_8U);
        
        // Clear the screen
        GLuint MatrixID = glGetUniformLocation(programID, "MVP");
        
        glm::tmat4x4<float> projection_matrix = pose.get_projection();
        projection_matrix[2][2] = projection_matrix[2][2]/viewport_height/viewport_width;
        glm::tmat4x4<float> viewport_matrix = get_viewport();
        glm::tmat4x4<float> MVP = projection_matrix * pose.get_modelview();
        GLuint TextureID = glGetUniformLocation(programID, "myTextureSampler");
        GLuint tex = matToTexture(isomap, GL_NEAREST, GL_NEAREST, GL_CLAMP);

        glBindTexture(GL_TEXTURE_2D, tex);

        glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
        glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec4) * vertices.size(), &vertices[0], GL_STATIC_DRAW);

        glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
        glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * texcoords.size(), &texcoords[0], GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0],
                     GL_STATIC_DRAW);

        // render target
        glBindFramebuffer(GL_FRAMEBUFFER, FramebufferID);
        glBindTexture(GL_TEXTURE_2D, renderedTexture);
        // Give an empty image to OpenGL ( the last "0" )
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, viewport_width, viewport_height, 0, GL_RGB, GL_UNSIGNED_BYTE,
                     0);

        // Poor filtering. Needed !
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        // The depth buffer
        glBindRenderbuffer(GL_RENDERBUFFER, depthrenderbuffer);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, viewport_width, viewport_height);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthrenderbuffer);

        // Set "renderedTexture" as our colour attachement #0
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, renderedTexture, 0);
        glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depthTexture, 0);

        // Set the list of draw buffers.
        GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
        glDrawBuffers(1, DrawBuffers); // "1" is the size of DrawBuffers
        assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        // Use our shader

        // std::cout << "glUseProgram(programID);"<< std::endl;

        // Send our transformation to the currently bound shader,
        // in the "MVP" uniform
        glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, isomapTexture);
        // Set our "myTextureSampler" sampler to use Texture Unit 0
        glUniform1i(TextureID, 0);

        // 1rst attribute buffer : vertices
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
        glVertexAttribPointer(
            0,        // attribute. No particular reason for 0, but must match the layout in the shader.
            4,        // size
            GL_FLOAT, // type
            GL_FALSE, // normalized?
            0,        // stride
            (void*)0  // array buffer offset
            );

        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
        glVertexAttribPointer(
            1,        // attribute. No particular reason for 1, but must match the layout in the shader.
            2,        // size
            GL_FLOAT, // type
            GL_FALSE, // normalized?
            0,        // stride
            (void*)0  // array buffer offset
            );

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer);

        // Draw the triangles !
        glDrawElements(GL_TRIANGLES,    // mode
                       indices.size(),  // count
                       GL_UNSIGNED_INT, // type
                       (void*)0         // element array buffer offset
                       );

        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);

        glBindFramebuffer(GL_FRAMEBUFFER, FramebufferID);
        glViewport(0, 0, viewport_width, viewport_height); // Render on the whole framebuffer, complete
        // // //use fast 4-byte alignment (default anyway) if possible
        glPixelStorei(GL_PACK_ALIGNMENT, (img.step & 3) ? 1 : 4);
        // // //set length of one complete row in destination data (doesn't need to equal img.cols)
        glPixelStorei(GL_PACK_ROW_LENGTH, img.step / img.elemSize());
        glReadPixels(0, 0, depth.cols, depth.rows, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, depth.data);
        glReadPixels(0, 0, img.cols, img.rows, GL_RGB, GL_UNSIGNED_BYTE, img.data);
        
        cv::flip(img, img, 0);
        cv::flip(depth, depth, 0);


        // Swap buffers
        glfwSwapBuffers(window);
        glfwPollEvents();

        return std::make_pair(img, depth);
    }

    void Update_isomap(cv::Mat new_isomap)
    {
        assert(isomap.cols == new_isomap.cols);
        assert(isomap.rows == new_isomap.rows);
        assert(isomap.type == new_isomap.type);
        glDeleteTextures(1, &isomapTexture);
        isomap = new_isomap; 
        isomapTexture = matToTexture(isomap, GL_NEAREST, GL_NEAREST, GL_CLAMP);
    }

    void terminate()
    {
        glDeleteBuffers(1, &vertexbuffer);
        glDeleteBuffers(1, &uvbuffer);
        glDeleteTextures(1, &isomapTexture);
        glDeleteProgram(programID);
        glDeleteVertexArrays(1, &VertexArrayID);
        // Close OpenGL window and terminate GLFW
        glfwTerminate();
    }

private:
    cv::Mat isomap;
    glm::tmat4x4<float> get_viewport()
    {
        glm::vec4 viewport = fitting::get_opencv_viewport(viewport_width, viewport_height);
        glm::mat4x4 viewport_matrix; // Identity matrix
        viewport_matrix[0][0] = 0.5f * viewport[2];
        viewport_matrix[3][0] = 0.5f * viewport[2] + viewport[0];
        viewport_matrix[1][1] = 0.5f * viewport[3];
        viewport_matrix[3][1] = 0.5f * viewport[3] + viewport[1];
        viewport_matrix[2][2] = 0.5f;
        viewport_matrix[3][2] = 0.5f;
        return viewport_matrix;
    }
} renderer;

} /* namespace render */
} /* namespace eos */

#endif /* RENDER_HPP_ */
