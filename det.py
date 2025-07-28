import numpy as np

values = [-2, -1, 0, 1, 2]
#values = [-1, 0, 1]

# Initialize a dictionary to store the matrices for each value of the determinant
det_matrices = {}
for det in range(0, 400):
    det_matrices[det] = []

# Loop over all possible 3x3 integer matrices
for i in values:
    for j in values:
        for k in values:
            for l in values:
                for m in values:
                    for n in values:
                        for o in values:
                            for p in values:
                                for q in values:
                                    # Construct the matrix
                                    A = np.array([[i, j, k], [l, m, n], [o, p, q]])
                                    # Calculate the absolute value of the determinant
                                    det = int(abs(np.linalg.det(A))+0.5)
                                    # Append the matrix to the list for the corresponding value of the determinant
                                    det_matrices[det].append(A)
for det in det_matrices:
    filename = f"det_{det}.txt"
    mats = det_matrices[det]
    if len(mats) != 0:
        with open(filename, 'w') as f:
            for matrix in mats:
                for i in range(3):
                    for j in range(3):
                        f.write(str(matrix[i,j])+" ")
                    f.write("\n")
