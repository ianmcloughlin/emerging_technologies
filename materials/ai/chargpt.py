import numpy as np

# Function to check if a pair of complex numbers form a valid qubit
def is_valid_qubit(alpha, beta):
    norm = np.abs(alpha)**2 + np.abs(beta)**2
    return np.isclose(norm, 1.0)

# Function to compute the tensor product of multiple qubits
def tensor_product(*qubits):
    result = qubits[0]
    for qubit in qubits[1:]:
        result = np.kron(result, qubit)
    return result

# Function to verify if a matrix is unitary
def is_unitary(matrix):
    identity = np.eye(matrix.shape[0])
    return np.allclose(matrix @ matrix.conj().T, identity)

# Function to apply a unitary matrix to a system of qubits
def apply_unitary(matrix, qubit_system):
    return np.dot(matrix, qubit_system)

# Example usage
# Define a valid qubit
alpha, beta = 1/np.sqrt(2), 1j/np.sqrt(2)
print("Is valid qubit:", is_valid_qubit(alpha, beta))

# Define qubit states |0> and |1>
qubit_0 = np.array([1, 0])
qubit_1 = np.array([0, 1])

# Compute tensor product of |0> and |1>
tensor_result = tensor_product(qubit_0, qubit_1)
print("Tensor product of |0> and |1>:", tensor_result)

# Define a unitary matrix (Hadamard gate)
H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]])
print("Is Hadamard gate unitary:", is_unitary(H))

# Apply Hadamard gate to |0>
applied_result = apply_unitary(H, qubit_0)
print("Result of applying Hadamard gate to |0>:", applied_result)



import sympy as sp

# Function to check if a pair of complex numbers form a valid qubit
def is_valid_qubit(alpha, beta):
    norm = sp.Abs(alpha)**2 + sp.Abs(beta)**2
    return sp.simplify(norm) == 1

# Function to compute the tensor product of multiple qubits
def tensor_product(*qubits):
    result = qubits[0]
    for qubit in qubits[1:]:
        result = sp.tensorproduct(result, qubit)
    return result

# Function to verify if a matrix is unitary
def is_unitary(matrix):
    identity = sp.eye(matrix.shape[0])
    return sp.simplify(matrix * matrix.H) == identity

# Function to apply a unitary matrix to a system of qubits
def apply_unitary(matrix, qubit_system):
    return matrix * qubit_system

# Example usage
# Define a valid qubit
alpha, beta = sp.Rational(1, sp.sqrt(2)), sp.I / sp.sqrt(2)
print("Is valid qubit:", is_valid_qubit(alpha, beta))

# Define qubit states |0> and |1>
qubit_0 = sp.Matrix([1, 0])
qubit_1 = sp.Matrix([0, 1])

# Compute tensor product of |0> and |1>
tensor_result = tensor_product(qubit_0, qubit_1)
print("Tensor product of |0> and |1>:", tensor_result)

# Define a unitary matrix (Hadamard gate)
H = (1/sp.sqrt(2)) * sp.Matrix([[1, 1], [1, -1]])
print("Is Hadamard gate unitary:", is_unitary(H))

# Apply Hadamard gate to |0>
applied_result = apply_unitary(H, qubit_0)
print("Result of applying Hadamard gate to |0>:", applied_result)
