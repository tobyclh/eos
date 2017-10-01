import eos
import dlib
import cv2
import os
import matplotlib.pyplot as plt

'''
In this tutorial we show that one can use BFM together with the cere fitting
'''
eos_python_dir = '/home/toby/Documents/toby-eos/python'
os.chdir(eos_python_dir)
img_path = '../share/obama.jpg'
img = cv2.imread(img_path)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../share/shape_predictor_68_face_landmarks.dat')
boxes = detector(img, 1)
print("Number of faces detected: {}".format(len(boxes)))
for i, d in enumerate(boxes):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        i, d.left(), d.top(), d.right(), d.bottom()))


shapes = [predictor(img, box) for box in boxes]
landmarks = [[[part.x, part.y] for part in shape.parts()]
            for shape in shapes][0]
for landmark in landmarks:
    print(*landmark)
    img = cv2.circle(img, (landmark[0], landmark[1]), 20, (0,0,255), -1)

# cv2.imshow('image', img)
plt.imshow(img)
plt.show()
# assert(False)

landmark_ids = list(map(str, range(1, 69)))
image_height = img.shape[0]
image_width = img.shape[1]

model = eos.morphablemodel.load_model("../share/bfm2009.bin")
blendshapes = eos.morphablemodel.load_blendshapes(
    "../share/bfm_expression.bin")
landmark_mapper = eos.core.LandmarkMapper('../share/ibug_bfm.txt')
edge_topology = eos.morphablemodel.load_edge_topology(
    '../share/bfm_edge_topology.json')
contour_landmarks = eos.fitting.ContourLandmarks.load(
    '../share/ibug_bfm.txt')
model_contour = eos.fitting.ModelContour.load('../share/model_contours_bfm.json')

#fit shape and pose for initial value
(mesh, pose, shape_coeffs, blendshape_coeffs) = eos.fitting.fit_shape_and_pose(model, blendshapes,
                                                                               landmarks, landmark_ids, landmark_mapper,
                                                                               image_width, image_height, edge_topology, contour_landmarks, model_contour)
eos.core.write_obj(mesh, 'obama.obj')


image = cv2.imread('../share/obama.jpg')
isomap = eos.render.extract_texture(mesh, pose,image, isomap_resolution=512)
cv2.imwrite('isomap.jpg')
