import eos
import dlib
import cv2
import os
'''
In this tutorial we show that one can use BFM together with the cere fitting
'''
eos_python_dir = ''
os.chdir(eos_python_dir)
img_path = '../share/obama.jpg'
img = cv2.imread(img_path)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../share/shape_predictor_68_face_landmarks.dat')
boxes = detector(img)
shapes = [shape_predictor(img, box) for box in boxes]
landmarks = [[[part.x, part.y] for part in shape.parts()]
            for shape in shapes][0]
landmark_ids = list(map(str, range(1, 69)))
image_height = img.shape[0]
image_width = img.shape[1]

model = eos.morphablemodel.load_model("../share/bfm.bin")
blendshapes = eos.morphablemodel.load_blendshapes(
    "../share/bfm_expression.bin")
landmark_mapper = eos.core.LandmarkMapper('../share/ibug_to_bfm.txt')
edge_topology = eos.morphablemodel.load_edge_topology(
    '../share/bfm_edge_topology.json')
contour_landmarks = eos.fitting.ContourLandmarks.load(
    '../share/ibug_to_bfm.txt')
model_contour = eos.fitting.ModelContour.load('../share/model_contours_bfm.json')

(mesh, pose, shape_coeffs, blendshape_coeffs) = eos.fitting.fit_shape_and_pose(model, blendshapes,
                                                                               landmarks, landmark_ids, landmark_mapper,
                                                                               image_width, image_height, edge_topology, contour_landmarks, model_contour)
# for BFM we cannot extract isomap unless you create a UV map for the model specifically.
eos.core.write_obj(mesh, 'simple_mesh.obj')

mesh, matrix1, matrix2, shape_coeffs, blendshape_coeffs, texture = eos.fitting.fit_shape_and_pose_ceres(
    model, blendshapes, landmarks, landmark_ids, landmark_mapper, img, contour_landmarks, model_contour)
eos.core.write_obj(mesh, 'mesh_cere.obj')

# print(type(mesh))
# print(dir(pose))
print(matrix1)
print(matrix2)
#%%
for arg in [mesh, matrix1, matrix2, image]:
    print(type(arg))
#%%
# isomap_cere = eos.render.extract_texture_v2(
#     mesh=mesh, view_model_matrix=matrix1, projection_matrix=matrix2, image=image, isomap_resolution=4096)
