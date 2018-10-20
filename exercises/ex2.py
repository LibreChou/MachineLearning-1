from library.PCA import *

def ex2():

    precision = .7

    # US Census
    input1 = [[1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000],
             [75.9950, 91.9720, 105.7110, 123.2030, 131.6690, 150.6970, 179.3230, 203.2120, 226.5050, 249.6330, 281.4220]]
    pca1 = PCA(input1)

    pca1.set_precision(precision)

    print("==== US Census ===")
    print("EigenVectors:")
    print(pca1.eigen_vectors)
    print("EigenValues:")
    print(pca1.eigen_values)
    print("Feature Vector:")
    print(pca1.feature_vector)
    print("Original data:")
    print(pca1.get_original_data())

    # Alps Water
    input2 = [[194.5, 194.3, 197.9, 198.4, 199.4, 199.9, 200.9, 201.1, 201.4, 201.3, 203.6, 204.6, 209.5, 208.6, 210.7, 211.9, 212.2],
             [20.79, 20.79, 22.4, 22.67, 23.15, 23.35, 23.89, 23.99, 24.02, 24.01, 25.14, 26.57, 28.49, 27.76, 29.04, 29.88, 30.06]]

    pca2 = PCA(input2)

    pca2.set_precision(precision)

    print("==== Alps Water ===")
    print("EigenVectors:")
    print(pca2.eigen_vectors)
    print("EigenValues:")
    print(pca2.eigen_values)
    print("Feature Vector:")
    print(pca2.feature_vector)
    print("Original data:")
    print(pca2.get_original_data())

    # Books x Grades
    input3 = [[0.0, 1.0, 0.0, 2.0, 4.0, 4.0, 1.0, 4.0, 3.0, 0.0, 2.0, 1.0, 4.0, 1.0, 0.0, 1.0, 3.0, 0.0, 1.0, 4.0, 4.0, 0.0, 2.0, 3.0, 1.0, 0.0, 3.0, 3.0, 2.0, 2.0, 3.0, 2.0, 2.0, 3.0, 4.0, 4.0, 3.0, 1.0, 2.0, 0.0],
             [9.0, 15.0, 10.0, 16.0, 10.0, 20.0, 11.0, 20.0, 15.0, 15.0, 8.0, 13.0, 18.0, 10.0, 8.0, 10.0, 16.0, 11.0, 19.0, 12.0, 11.0, 19.0, 15.0, 15.0, 20.0, 6.0, 15.0, 19.0, 14.0, 13.0, 17.0, 20.0, 11.0, 20.0, 20.0, 20.0, 9.0, 8.0, 16.0, 10.0],
             [45.00, 57.00, 45.00, 51.00, 65.00, 88.00, 44.00, 87.00, 89.00, 59.00, 66.00, 65.00, 56.00, 47.00, 66.00, 41.00, 56.00, 37.00, 45.00, 58.00, 47.00, 64.00, 97.00, 55.00, 51.00, 61.00, 69.00, 79.00, 71.00, 62.00, 87.00, 54.00, 43.00, 92.00, 83.00, 94.00, 60.00, 56.00, 88.00, 62.00]]

    pca3 = PCA(input3)

    pca3.set_precision(precision)

    print("==== PCA ===")
    print("EigenVectors:")
    print(pca3.eigen_vectors)
    print("EigenValues:")
    print(pca3.eigen_values)
    print("Feature Vector:")
    print(pca3.feature_vector)
    print("Original data:")
    print(pca3.get_original_data())



