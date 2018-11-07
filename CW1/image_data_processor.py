import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random


# Compute number of images per person (assuming each face has same num of images)
def images_per_person(results):
    images_per_face = 0
    image_count = 0
    while images_per_face == 0:
        if results[image_count] == results[image_count + 1]:
            image_count += 1
        else:
            images_per_face = image_count + 1
    return images_per_face


# Compute number of distinct faces in the data
def distinct_faces_num(total_faces_num, images_per_face):
    if total_faces_num % images_per_face != 0:
        raise ValueError("Have different number of image per face")
    else:
        return int(total_faces_num / images_per_face)


# Split training samples and test samples
def split_train_test(total_faces_num,
                     test_image_per_face,
                     images_per_face,
                     num_of_distinct_faces,
                     resolutions,
                     faces,
                     results):

    # Calculating the number of images, setting up for the training and test sample split
    train_image_per_face = images_per_face - test_image_per_face

    num_of_train_faces = train_image_per_face * num_of_distinct_faces
    num_of_test_faces = test_image_per_face * num_of_distinct_faces

    test_ratio = num_of_test_faces / total_faces_num

    faces_train = np.zeros((resolutions, num_of_train_faces))
    faces_test = np.zeros((resolutions, num_of_test_faces))
    results_train = np.zeros(num_of_train_faces)
    results_test = np.zeros(num_of_test_faces)

    # Split training samples and test samples
    for i in range(0, num_of_distinct_faces):
        state = random.randint(1, 100)
        start = i * images_per_face
        end = start + images_per_face
        single_face_arr = faces[:, start: end]
        single_face_result = results[start: end]
        faces_train_temp, faces_test_temp, results_train_temp, results_test_temp = train_test_split(single_face_arr.transpose(),
                                                                                                    single_face_result,
                                                                                                    test_size=test_ratio,
                                                                                                    random_state=state)
        start_train = start if start == 0 else start - test_image_per_face * i
        end_train = end - test_image_per_face * (i + 1)
        start_test = start if start == 0 else start - train_image_per_face * i
        end_test = end - train_image_per_face * (i + 1)
        faces_train[:, start_train: end_train] = faces_train_temp.transpose()
        faces_test[:, start_test: end_test] = faces_test_temp.transpose()
        results_train[start_train: end_train] = results_train_temp
        results_test[start_test: end_test] = results_test_temp

    return num_of_train_faces, num_of_test_faces, faces_train, faces_test, results_train, results_test


def normalization(vector):
    norm = np.linalg.norm(vector)
    vector /= norm


def print_image(image):
    # Rescale average training vector to gray scale matrix
    image_to_print = image.reshape(46, 56)

    # Show image
    image_to_print = image_to_print.T
    plt.imshow(image_to_print, cmap='gist_gray')
    plt.show()


def sample_reconstruction(num_of_faces, projections, resolutions, best_eigen_vectors, face_avg, M):
    train_faces_reconstructed = np.zeros((resolutions, num_of_faces), dtype=np.complex)

    # Reconstruct training faces as linear combination of the best M eigen vectors
    for i in range(0, projections.shape[0]):
        linear_combination_of_eigen_vectors = np.zeros((resolutions, 1), dtype=np.complex)
        for j in range(0, M):
            projection = projections[i][j]
            eigen_vector = best_eigen_vectors[j].reshape(resolutions, 1)
            linear_combination_of_eigen_vectors += eigen_vector * projection
        train_faces_reconstructed[:, i] = (face_avg + linear_combination_of_eigen_vectors).squeeze()

    return train_faces_reconstructed


def image_comparison(pca):
    first_reconstructed_image = pca.test_sample_reconstructed[:, 0].real.reshape(46, 56).T
    second_reconstructed_image = pca.test_sample_reconstructed[:, 1].real.reshape(46, 56).T
    third_reconstructed_image = pca.test_sample_reconstructed[:, 2].real.reshape(46, 56).T
    first_test_image = pca.test_sample[:, 0].reshape(46, 56).T
    second_test_image = pca.test_sample[:, 1].reshape(46, 56).T
    third_test_image = pca.test_sample[:, 2].reshape(46, 56).T

    plt.subplot(321)
    plt.title('Actual')
    plt.imshow(first_test_image, cmap='gist_gray')
    plt.subplot(323)
    plt.imshow(second_test_image, cmap='gist_gray')
    plt.subplot(325)
    plt.imshow(third_test_image, cmap='gist_gray')
    plt.subplot(322)
    plt.title('Reconstructed')
    plt.imshow(first_reconstructed_image, cmap='gist_gray')
    plt.subplot(324)
    plt.imshow(second_reconstructed_image, cmap='gist_gray')
    plt.subplot(326)
    plt.imshow(third_reconstructed_image, cmap='gist_gray')
    plt.show()