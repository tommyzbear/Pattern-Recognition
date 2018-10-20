from mat4py import loadmat
import numpy as np
from sklearn.model_selection import train_test_split

data = loadmat('face(1).mat')

for key in data:
    print(key)

faces = data.get("X")
results = data.get("l")

faces_arr = np.asarray(faces)
results_arr = np.asarray(results)

faces_arr = faces_arr.transpose()

image_pixels = faces_arr.shape[-1]

#print(faces_arr.shape, results_arr.shape)

#print(type(faces_arr), type(results_arr))

faces_train, faces_test, results_train, results_test = train_test_split(faces_arr, results_arr, test_size=0.3, random_state=10)

#print(faces_train.shape, faces_test.shape)

face_sum = np.zeros((1, image_pixels))

faces_train_num = faces_train.shape[0]

for face in range(faces_train_num):
    face_sum += faces_train[face]

face_avg = face_sum / faces_train_num

#print(face_avg)

faces_train_covariance = 1 / faces_train_num
phi_dot_product_sum = np.zeros((image_pixels, image_pixels))

for face in range(faces_train_num):
    phi = faces_train[face] - face_avg
    phi_dot_product = phi.transpose().dot(phi)
    phi_dot_product_sum += phi_dot_product

faces_train_covariance *= phi_dot_product_sum

#print(faces_train_covariance.shape, '\n', faces_train_covariance)

eigen_values, eigen_vectors = np.linalg.eig(faces_train_covariance)

print(eigen_values, "\n", eigen_vectors)
