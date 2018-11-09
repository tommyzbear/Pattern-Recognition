from pca import *
import numpy as np

class PCA_LDA:
    '''Fisherface'''
    def __init__(self,
                 test_samples,
                 train_samples,
                 train_results,
                 num_of_test_samples,
                 num_of_train_samples,
                 num_of_distinct_samples,
                 resolution,
                 M_pca,
                 M_lda):
        self.train_samples = train_samples
        self.test_sample = test_samples
        self.train_sample = train_samples
        self.train_results = train_results
        self.num_of_test_samples = num_of_test_samples
        self.num_of_train_samples = num_of_train_samples
        self.num_of_distinct_samples = num_of_distinct_samples
        self.resolution = resolution
        self.M_pca = M_pca
        self.M_lda = M_lda
        self.train_avg_vector = avg_face_vector(train_samples,
                                                self.resolution,
                                                self.num_of_train_samples).reshape(self.resolution, 1)
        self.opt_eig_vec = None
        self.train_sample_projection = None
        self.test_sample_projection = None

    def fit(self):
        train_sample_per_class = int(self.train_samples.shape[-1] / self.num_of_distinct_samples)
        class_mean = np.zeros((self.resolution, self.num_of_distinct_samples))
        for i in range(0, class_mean.shape[-1]):
            class_mean[:, i] = self.train_samples[:,
                               i * train_sample_per_class: (i + 1) * train_sample_per_class].mean(1)

        class_normalized_mean = class_mean - self.train_avg_vector

        # Between-class scatter matrix
        S_B = class_normalized_mean @ class_normalized_mean.T

        # Compute x - mi
        discriminant_train_samples = np.zeros((self.train_samples.shape[0], self.train_samples.shape[-1]))
        index = 0
        for i in range(0, self.train_samples.shape[-1]):
            discriminant_train_samples[:, i] = self.train_samples[:, i] - class_normalized_mean[:, index]
            if (i + 1) % train_sample_per_class == 0:
                index += 1

        # Within-class scatter matrix
        S_W = discriminant_train_samples @ discriminant_train_samples.T

        # Get low-dimension PCA training projections
        pca = PCA(self.test_sample,
                  self.train_samples,
                  self.train_results,
                  self.num_of_test_samples,
                  self.num_of_train_samples,
                  self.resolution,
                  self.M_pca,
                  True)

        pca.projection()
        pca_best_eig_vec = pca.dimensioned_eig_vectors

        # Compute generalized eigen vectors and eigen values
        numerator = pca_best_eig_vec.transpose() @ S_B @ pca_best_eig_vec
        denominator = pca_best_eig_vec.transpose() @ S_W @ pca_best_eig_vec

        lda_eig_val, lda_eig_vec = np.linalg.eig(numerator @ np.linalg.inv(denominator))

        # numerator = abs(generalized_eig_vec.transpose() @ pca_best_eig_vec.transpose() @ S_B @ pca_best_eig_vec @ generalized_eig_vec)
        # denominator = abs(generalized_eig_vec.transpose() @ pca_best_eig_vec.transpose() @ S_W @ pca_best_eig_vec @ generalized_eig_vec)

        # Retrieve largest M eigen value indices in the array
        largest_eig_value_indices = np.argsort(lda_eig_val)[-self.M_lda:]

        # Initialize best eigen vectors
        best_lda_eig_vec = np.zeros((lda_eig_vec.shape[0], self.M_lda), dtype=np.complex)

        # Retrieve corresponding eigen vectors mapping to top M eigen values
        for i in range(0, self.M_lda):
            best_lda_eig_vec[:, i] = lda_eig_vec[:, largest_eig_value_indices[i]]

        self.opt_eig_vec = best_lda_eig_vec.transpose() @ pca_best_eig_vec.transpose()

        # normalize training samples
        normalized_train_samples = self.train_sample - self.train_avg_vector

        # normalize test samples
        normalized_test_samples = self.test_sample - self.train_avg_vector

        self.train_sample_projection = normalized_train_samples.transpose() @ self.opt_eig_vec.transpose()
        self.test_sample_projection = normalized_test_samples.transpose() @ self.opt_eig_vec.transpose()


def avg_face_vector(samples, resolution, num_of_samples):
    faces_sum = np.zeros(resolution)

    for i in range(num_of_samples):
        temp = samples[:, i]
        faces_sum += temp

    return faces_sum / num_of_samples
