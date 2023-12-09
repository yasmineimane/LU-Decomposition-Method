import numpy as np

"""
    get_num - Prompt the user to enter the number of variables in the system.

    Returns:
    int: Number of variables in a system.
"""
def get_num():
    num = int(input("Enter the number of variables in the system : "))
    return num


"""
    get_coefficients - get coefficients of the system linear frm the use input.

    Args:
    num (int): number of variables in the system.

    Returns:
    np.ndarray: A NumPy array representing the filled square matrix.
"""
def get_coefficients(num):
    coefficients = []

    print(f"Enter coefficients for the system of equation whith {num} variables")
    for i in range(num):
        coefficient = [int(input(f"Enter the coefficient {i + 1} for variable {j + 1} : ")) for j in range(num)]
        coefficients.append(coefficient)

    square_matrix = np.array(coefficients)
    return square_matrix

"""
    get_constants - get constant of the system linear from the user input.

    Args:
    num (int): number of variables in the system.

    Returns:
    np.ndarray: A NumPy array representing the vector of constants.
"""
def get_constants(num):
    constants = []

    for i in range(num):
        constant = int (input(f"Enter the constant term for equation {i + 1} : "))
        constants.append(constant)
    constants = np.array(constants)

    return constants


"""
    LU - Perform LU decomposition on a square matrix.

    Parameters:
    M (np.ndarray): Square matrix to decompose.

    Returns:
    Tuple[np.ndarray, np.ndarray]: Tuple containing the lower triangular matrix (L) and the upper triangular matrix (U).
"""
def LU(M):
    n = M.shape[0]
    U = np.copy(M)
    L = np.eye(n)

    for i in range(n):
        p = U[i,i]
        
        for j in range(i + 1, n):
            L[j, i] = U[j, i] / p
            U[j] = U[j] - L[j, i] * U[i]
    
    L = np.round(L, decimals=2)
    
    return L, U


"""
    solve_linear_system - Solve a system of linear equations using LU decomposition.

    Parameters:
    L (np.ndarray): Lower triangular matrix from LU decomposition.
    U (np.ndarray): Upper triangular matrix from LU decomposition.
    constants (np.ndarray): Vector on the right-hand side of the linear system.

    Returns:
    np.ndarray: Solution vector for the linear system.
"""
def solve_linear_system(L, U, constants):
    if L.ndim < 2 or U.ndim < 2 or constants.ndim < 1:
        raise ValueError("Input arrays must have at least 2 dimensions (matrix/vector).")
    n = len(L)
    if L.shape[0] != n or L.shape[1] != n or U.shape[0] != n or U.shape[1] != n or constants.shape[0] != n:
        raise ValueError("Input array shapes are incompatible with the size of the linear system.")
    y = np.zeros(n)
    x = np.zeros(n)
    for i in range(n):
        y[i] = constants[i]
        for j in range(i):
            y[i] = y[i] - L[i][j] * y[j]
    for i in range(n - 1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, n):
            x[i] = x[i] - U[i, j] * x[j]
        
        if U[i, i] == 0:
            x[i] = 0
        else:
            x[i] = x[i] / U[i, i]
    x = np.round(x).astype(int)

    return x


"""
    FactLU - Main function.

    Parameters:
    No parameters.
"""
def FactLU():
    num = get_num()
    A = get_coefficients(num)
    det = np.linalg.det(A)
    if det == 0:
        raise ValueError("LU can't be achieved try again with another values.")
    else:
        B = get_constants(num)
        L, U = LU(A)
        solution = solve_linear_system(L, U, B)

        print("Coefficients matrix A:")
        print(A)
        print("\nConstants vector B:")
        print(B.reshape(-1, 1))
        print("\nLower triangular matrix L:")
        print(L)
        print("\nUpper triangular matrix U:")
        print(U)
        print("\nSolution vector:")
        print(solution.reshape(-1, 1))



# Main Program
FactLU()