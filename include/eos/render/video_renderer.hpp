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

#include <Eigen/Core>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
namespace eos {
namespace render {

class Viewer
{
public:
    std::vector<glm::vec2> frame_vertices;
    std::vector<glm::vec2> reenacted_vertices;
    std::vector<unsigned int> indices; /// vertex index
    GLfloat scale;
    glm::vec2 normalized_offsets;
    glm::vec2 translation;
    cv::Mat mvp;
    
    int viewport_width;
    int viewport_height;
    cv::Mat frame; // the canvas we will be rendering on, constant for video -> photo

   // OpenGL handlers
    GLFWwindow* window;
    GLuint MatrixID;
    GLuint programID;
    GLuint TextureID;
    GLuint RenderbufferID;
    GLuint imgvertexbuffer;
    GLuint reenactedvertexbuffer;
    GLuint elementbuffer;
    GLuint depthrenderbuffer;
    GLuint frameTexture;
    GLuint renderedTexture;
    GLuint depthTexture;

    //background handlers
    GLuint backgroundProgramID;
    GLuint backgroundID;
    GLuint backgroundVertex;
    GLuint backgroundUV;
    GLuint backgroundTexture;    

    Viewer(){};

    Viewer(std::vector<glm::vec3> _vertices, std::vector<std::array<int, 3>> tvi, fitting::RenderingParameters _pose, cv::Mat _view)
        : frame_vertices(_vertices), frame(_view), pose(_pose)
    {
        for (int i = 0; i < tvi.size(); i++)
        {
            for (int j = 0; j < 3; j++)
            {
                unsigned int vertexIndex = (unsigned int)(tvi.at(i)[j]);
                indices.push_back(vertexIndex);
            }
        }
        reenacted_vertices = _vertices;

        viewport_width = _view.cols;
        viewport_height = _view.rows;

        if (!glfwInit())
        {
            fprintf(stderr, "Failed to initialize GLFW\n");
            getchar();
            return;
        }
        std::cout << "Apple" << std::endl;
        glfwWindowHint(GLFW_SAMPLES, 8);
        glfwWindowHint(GLFW_OPENGL_ANY_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        window = glfwCreateWindow(viewport_width, viewport_height, "Viewer", NULL, NULL);
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
        // std::cout << "Banana" << std::endl;
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
        // glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

        // // Enable depth test
        glEnable(GL_DEPTH_TEST);
        // glEnable(GL_DEPTH_CLAMP);

        // glDepthRange(-100, 10);
        // Accept fragment if it closer to the camera than the former one
        glDepthFunc(GL_LESS);
        glShadeModel(GL_SMOOTH);

        // std::cout << "Pencil" << std::endl;
        programID = LoadShaders("eos_vertex.vert", "eos_fragment.frag");
        std::cout << "Loaded eos_vertex.vert" << std::endl;

        glGenBuffers(1, &imgvertexbuffer);
        glGenBuffers(1, &reenactedvertexbuffer);
        glGenBuffers(1, &elementbuffer);
        glGenFramebuffers(1, &RenderbufferID);
        glGenRenderbuffers(1, &depthrenderbuffer);
        glGenTextures(1, &renderedTexture);
        glGenTextures(1, &depthTexture);
        glGenTextures(1, &frameTexture);
        glGenTextures(1, &backgroundTexture);

        glUseProgram(programID);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0],
                     GL_STATIC_DRAW);

        // render target
        glBindFramebuffer(GL_FRAMEBUFFER, RenderbufferID);
        glBindTexture(GL_TEXTURE_2D, renderedTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, viewport_width, viewport_height, 0, GL_RGB, GL_UNSIGNED_BYTE,
                     0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        // The depth buffer
        glBindRenderbuffer(GL_RENDERBUFFER, depthrenderbuffer);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, viewport_width, viewport_height);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthrenderbuffer);

        // Set "renderedTexture" as our colour attachement #0
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, renderedTexture, 0);
        glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depthTexture, 0);
        TextureID = glGetUniformLocation(programID, "myTextureSampler");

        // set up background texture
        cv::flip(frame, frame, 0);        
        backgroundProgramID = LoadShaders("background_vertex.vert", "background_fragment.frag");
        std::cout << "Loaded background_vertex.vert" << std::endl;
        glUseProgram(backgroundProgramID);
        matToTexture(frame, GL_NEAREST, GL_NEAREST, GL_CLAMP, backgroundTexture);
        backgroundID = glGetUniformLocation(backgroundProgramID, "myBackground");
        static const GLfloat g_vertex_buffer_data[] = { 
            -1.0f,-1.0f,
            -1.0f, 1.0f,
            1.0f, -1.0f,
            1.0f, 1.0f,
            1.0f,-1.0f,
            -1.0f, 1.0f
        };
        static const GLfloat g_uv_data[] = { 
            0.0f,0.0f,
            0.0f,1.0f,
            1.0f, 0.0f,
            1.0f, 1.0f,
            1.0f,0.0f,
            0.0f, 1.0f
        };
        glGenTextures(1, &backgroundVertex);
        glGenTextures(1, &backgroundUV);
        glBindBuffer(GL_ARRAY_BUFFER, backgroundVertex);
        glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), &g_vertex_buffer_data[0], GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, backgroundUV);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(g_uv_data), &g_uv_data[0],GL_STATIC_DRAW);

        
        return;
    }

    std::pair<cv::Mat, cv::Mat> render()
    {
        cv::Mat img(viewport_height, viewport_width, CV_8UC3);
        cv::Mat depth(viewport_height, viewport_width, CV_8U);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        {
            glUseProgram(backgroundProgramID);        
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, backgroundTexture);
            glUniform1i(backgroundID, 0);
            glEnableVertexAttribArray(0);
            glBindBuffer(GL_ARRAY_BUFFER, backgroundVertex);
            glVertexAttribPointer(
                0,        // attribute. No particular reason for 0, but must match the layout in the shader.
                2,        // size
                GL_FLOAT, // type
                GL_FALSE, // normalized?
                0,        // stride
                (void*)0  // array buffer offset
                );
            glEnableVertexAttribArray(1);
            glBindBuffer(GL_ARRAY_BUFFER, backgroundUV);
            glVertexAttribPointer(
                1,        // attribute. No particular reason for 1, but must match the layout in the shader.
                2,        // size
                GL_FLOAT, // type
                GL_FALSE, // normalized?
                0,        // stride
                (void*)0  // array buffer offset
                );
            glDrawArrays(GL_TRIANGLES, 0, 6);
        }

        //draw foreground
        {
            glUseProgram(programID);
            GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
            glDrawBuffers(1, DrawBuffers); // "1" is the size of DrawBuffers
            assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE);
                      
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, frameTexture);
            // std::cout << "a" << std::endl;
            // Set our "myTextureSampler" sampler to use Texture Unit 0
            glUniform1i(TextureID, 0);
            // 1rst attribute buffer : vertices
            glEnableVertexAttribArray(0);
            glBindBuffer(GL_ARRAY_BUFFER, imgvertexbuffer);
            glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * frame_vertices.size(),
                        &frame_vertices[0], GL_STREAM_DRAW);
            glVertexAttribPointer(
                0,        // attribute. No particular reason for 0, but must match the layout in the shader.
                2,        // size
                GL_FLOAT, // type
                GL_FALSE, // normalized?
                0,        // stride
                (void*)0  // array buffer offset
            );
            // std::cout << "a" << std::endl;
            glEnableVertexAttribArray(1);
            glBindBuffer(GL_ARRAY_BUFFER, reenactedvertexbuffer);
            glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * reenacted_vertices.size(),
                         &reenacted_vertices[0], GL_STREAM_DRAW);
            glVertexAttribPointer(
                1,        // attribute. No particular reason for 0, but must match the layout in the shader.
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
        }
        // std::cout << "a" << std::endl;

        glBindFramebuffer(GL_FRAMEBUFFER, RenderbufferID);
        glViewport(0, 0, viewport_width, viewport_height); // Render on the whole framebuffer, complete
        // // //use fast 4-byte alignment (default anyway) if possible
        glPixelStorei(GL_PACK_ALIGNMENT, (img.step & 3) ? 1 : 4);
        // // //set length of one complete row in destination data (doesn't need to equal img.cols)
        glPixelStorei(GL_PACK_ROW_LENGTH, img.step / img.elemSize());
        glReadPixels(0, 0, depth.cols, depth.rows, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, depth.data);
        glReadPixels(0, 0, img.cols, img.rows, GL_RGB, GL_UNSIGNED_BYTE, img.data);
        // std::cout << "c" << std::endl;
        cv::flip(img, img, 0);
        cv::flip(depth, depth, 0);

        // Swap buffers
        glfwSwapBuffers(window);
        glfwPollEvents();
        // std::cout << "b" << std::endl;
        return std::make_pair(img, depth);
    }

    void Update_frame(cv::Mat _frame)
    {
        cv::Mat cFrame;
        cv::flip(_frame, cFrame, 0);
        matToTexture(cFrame, GL_NEAREST, GL_NEAREST, GL_CLAMP, frameTexture);
        matToTexture(cFrame, GL_NEAREST, GL_NEAREST, GL_CLAMP, backgroundTexture);
    }

    void Set_pose(eos::fitting::RenderingParameters _pose)
    {
        // pose = _pose;
    }

    void terminate()
    {
        glDeleteBuffers(1, &imgvertexbuffer);
        glDeleteProgram(programID);
        glfwTerminate();
    }

private:
    int debugMode = 1;
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