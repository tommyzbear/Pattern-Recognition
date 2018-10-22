import numpy as np
import image_data_processor as idp
from eigen_face import *

# This class is for initializing for classification and applying different classification methods
class PRFactory:
    def __init__(self, faces, result, test_image_per_face):
        self.faces = np.asarray(faces).transpose()
        self.result = np.asarray(result)
        self.test_image_per_face = test_image_per_face
        self.resolutions = self.faces.shape[-1]
        self.total_faces_num = self.faces.shape[0]
        self.images_per_face = idp.images_per_person(self.result)
        self.num_of_distinct_face = idp.distinct_faces_num(self.total_faces_num, self.images_per_face)
        self.num_of_train_samples, self.num_of_test_samples, self.train_samples, self.test_samples, self.train_results, self.test_results = idp.split_train_test(
            self.total_faces_num,
            test_image_per_face,
            self.images_per_face,
            self.num_of_distinct_face,
            self.resolutions,
            self.faces,
            self.result)
            
    # Compute learning result by using Nearest Neighbour classification
    def nearest_neighbour(self):
        # Initialize learning_results
        learning_result = np.zeros(self.num_of_test_samples)

        i = 0

        # Compute projections of training faces onto eigen space
        train_eigen_faces = EigenFace(self.train_samples, self.resolutions, self.num_of_train_samples)
        projections_of_train_faces = train_eigen_faces.projections_of_faces

        # Normalized test samples by subtracting average training face vector
        normalized_faces_test = self.test_samples - train_eigen_faces.face_avg_vector

        # Compute projections of testing faces onto eigen space
        projections_of_test_faces = np.matmul(normalized_faces_test, train_eigen_faces.best_eigen_vectors.transpose())

        # Compute learning result by using Nearest Neighbour classification
        for test_projection in projections_of_test_faces:
            error = np.zeros((projections_of_train_faces.shape[0]))
            index = 0
            for train_projection in projections_of_train_faces:
                error[index] = np.linalg.norm(test_projection - train_projection)
                index += 1
            learning_result[i] = self.train_results[np.argmin(error)]
            i += 1

        return learning_result
