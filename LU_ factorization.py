import numpy as np

from colors import bcolors
from matrix_utility import swap_rows_elementary_matrix, row_addition_elementary_matrix

def swap_rows(A):
    A = np.array(A)
    for i in range(A.shape[0]):
        if A[i][i]==0 and i+1 < A.shape[0] and A[i+1][i]!=0:
            A[[i,i+1]] = A[[i+1,i]]
    return A

def lu(A):
    N = len(A)
    L = np.eye(N) # Create an identity matrix of size N x N

    for i in range(N):
        # Check if the diagonal element is zero
        if A[i][i] == 0:
            A = swap_rows(A)  # Swap rows to avoid division by zero
            if A[i][i] == 0:  # Check again after swapping
                raise ValueError("Matrix is singular and cannot be decomposed.")

        # Partial Pivoting: Find the pivot row with the largest absolute value in the current column
        pivot_row = i
        v_max = A[pivot_row][i]
        for j in range(i + 1, N):
            if abs(A[j][i]) > v_max:
                v_max = A[j][i]
                pivot_row = j

        # Swap the current row with the pivot row
        if pivot_row != i:
            e_matrix = swap_rows_elementary_matrix(N, i, pivot_row)
            #print(f"elementary matrix for swap between row {i} to row {pivot_row} :\n {e_matrix} \n")
            A = np.dot(e_matrix, A)
            #print(f"The matrix after elementary operation :\n {A}")
            #print(bcolors.OKGREEN,"---------------------------------------------------------------------------", bcolors.ENDC)

        for j in range(i + 1, N):
            # Compute the multiplier
            m = -A[j][i] / A[i][i]
            e_matrix = row_addition_elementary_matrix(N, j, i, m)
            e_inverse = np.linalg.inv(e_matrix)
            L = np.dot(L, e_inverse)
            A = np.dot(e_matrix, A)
            #print(f"elementary matrix to zero the element in row {j} below the pivot in column {i} :\n {e_matrix} \n")
            #print(f"The matrix after elementary operation :\n {A}")
            #print(bcolors.OKGREEN,"---------------------------------------------------------------------------", bcolors.ENDC)

    U = A
    return L, U



# function to calculate the values of the unknowns
def backward_substitution(mat):
    N = len(mat)
    x = np.zeros(N)  # An array to store solution

    # Start calculating from last equation up to the first
    for i in range(N - 1, -1, -1):

        x[i] = mat[i][N]

        # Initialize j to i+1 since matrix is upper triangular
        for j in range(i + 1, N):
            x[i] -= mat[i][j] * x[j]

        x[i] = (x[i] / mat[i][i])

    return x

def lu_solve(A_b):
    L, U = lu(A_b)
    #print("Lower triangular matrix L:\n", L)
    #print("Upper triangular matrix U:\n", U[:, :-1])

    result = backward_substitution(U)
    print(bcolors.OKBLUE,"\nSolution for the system:")
    for x in result:
        print("{:.6f}".format(x))


if __name__ == '__main__':


    A_b=[[-1, 1, 3, -3, 1, -1],
         [3, -3, -4, 2, 3, 18],
         [2, 1, -5, -3, 5, 6],
         [-5, -6, 4, 1, 3, 22],
         [3, -2, -2, -3, 5, 10]]
    print("Original matrix A:")
    for row in A_b:
        print(row)

    lu_solve(A_b)