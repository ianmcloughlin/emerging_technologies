# Modeling Qubits with NumPy and SymPy

# NUMPY

## Introduction

This notebook demonstrates how to model quantum bits (qubits) using both NumPy and SymPy. We'll cover:

1. Checking if a pair of complex numbers forms a valid qubit
2. Calculating tensor products of multiple qubits
3. Verifying if a matrix is unitary
4. Calculating the effect of a unitary matrix on a system of qubits

We'll first implement these operations using NumPy, then using SymPy, and finally compare the two approaches.

## Part 1: Qubit Modeling with NumPy

Let's start by importing NumPy and defining some helper functions.

```python
import numpy as np
```

### 1.1 Checking if a Complex Vector Forms a Valid Qubit

A qubit is represented by a 2D complex vector $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$ where $\alpha$ and $\beta$ are complex numbers that satisfy $|\alpha|^2 + |\beta|^2 = 1$.

```python
def is_valid_qubit_numpy(state_vector):
    """
    Check if a vector represents a valid qubit state.

    Args:
        state_vector: A numpy array of shape (2,) containing complex numbers

    Returns:
        bool: True if the state is a valid qubit, False otherwise
    """
    # Ensure the vector has the right shape
    if state_vector.shape != (2,):
        return False

    # Calculate the norm and check if it's approximately 1
    norm = np.sum(np.abs(state_vector)**2)
    return np.isclose(norm, 1.0)

# Examples
print("Example 1: |0⟩ state")
state_0 = np.array([1+0j, 0+0j])
print(f"Is valid qubit: {is_valid_qubit_numpy(state_0)}")

print("\nExample 2: |1⟩ state")
state_1 = np.array([0+0j, 1+0j])
print(f"Is valid qubit: {is_valid_qubit_numpy(state_1)}")

print("\nExample 3: |+⟩ state")
state_plus = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
print(f"Is valid qubit: {is_valid_qubit_numpy(state_plus)}")

print("\nExample 4: Invalid state (norm != 1)")
invalid_state = np.array([0.5+0j, 0.5+0j])
print(f"Is valid qubit: {is_valid_qubit_numpy(invalid_state)}")
```

### 1.2 Calculating Tensor Products of Multiple Qubits

The tensor product (Kronecker product) of two qubits creates a 4-dimensional vector representing a 2-qubit system.

```python
def tensor_product_numpy(qubit_list):
    """
    Calculate the tensor product of multiple qubit states.

    Args:
        qubit_list: List of numpy arrays, each representing a qubit state

    Returns:
        numpy.ndarray: Tensor product of all qubits
    """
    result = qubit_list[0]
    for qubit in qubit_list[1:]:
        result = np.kron(result, qubit)
    return result

# Example
print("Tensor product examples:")
print("Tensor product of |0⟩ ⊗ |0⟩:")
tensor_00 = tensor_product_numpy([state_0, state_0])
print(tensor_00)
print("\nTensor product of |0⟩ ⊗ |1⟩:")
tensor_01 = tensor_product_numpy([state_0, state_1])
print(tensor_01)
print("\nTensor product of |+⟩ ⊗ |+⟩:")
tensor_plusplus = tensor_product_numpy([state_plus, state_plus])
print(tensor_plusplus)
```

### 1.3 Verifying if a Matrix is Unitary

A matrix U is unitary if U†U = I, where U† is the conjugate transpose of U and I is the identity matrix.

```python
def is_unitary_numpy(matrix):
    """
    Check if a matrix is unitary.

    Args:
        matrix: A square numpy array

    Returns:
        bool: True if the matrix is unitary, False otherwise
    """
    if matrix.shape[0] != matrix.shape[1]:
        return False

    # Calculate U†U
    product = np.dot(matrix.conj().T, matrix)

    # Check if U†U is approximately the identity matrix
    identity = np.eye(matrix.shape[0])
    return np.allclose(product, identity)

# Define some common quantum gates
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)  # Hadamard gate
X = np.array([[0, 1], [1, 0]])                # Pauli-X gate
Y = np.array([[0, -1j], [1j, 0]])             # Pauli-Y gate
Z = np.array([[1, 0], [0, -1]])               # Pauli-Z gate
I = np.eye(2)                                 # Identity gate

print("Check if common quantum gates are unitary:")
print(f"Hadamard (H): {is_unitary_numpy(H)}")
print(f"Pauli-X: {is_unitary_numpy(X)}")
print(f"Pauli-Y: {is_unitary_numpy(Y)}")
print(f"Pauli-Z: {is_unitary_numpy(Z)}")
print(f"Identity: {is_unitary_numpy(I)}")

# Create a non-unitary matrix for comparison
non_unitary = np.array([[1, 1], [1, 1]])
print(f"Non-unitary matrix: {is_unitary_numpy(non_unitary)}")
```

### 1.4 Calculating the Effect of a Unitary Matrix on Qubits

To apply a unitary operation to a qubit system, we multiply the state vector by the unitary matrix.

```python
def apply_unitary_numpy(unitary, state):
    """
    Apply a unitary operation to a quantum state.

    Args:
        unitary: A unitary matrix as numpy array
        state: A quantum state vector as numpy array

    Returns:
        numpy.ndarray: The resulting quantum state
    """
    return np.dot(unitary, state)

# Examples
print("Applying unitary operations to qubit states:")

print("\nApplying H to |0⟩:")
result_H0 = apply_unitary_numpy(H, state_0)
print(result_H0)
print(f"This is the |+⟩ state. Verification: {np.allclose(result_H0, state_plus)}")

print("\nApplying X (NOT gate) to |0⟩:")
result_X0 = apply_unitary_numpy(X, state_0)
print(result_X0)
print(f"This is the |1⟩ state. Verification: {np.allclose(result_X0, state_1)}")

print("\nApplying Z to |+⟩:")
result_Zplus = apply_unitary_numpy(Z, state_plus)
print(result_Zplus)
```

### 1.5 Multiple Qubit Operations

For multi-qubit systems, we need to use the tensor product of operators to apply gates to specific qubits.

```python
def apply_gate_to_qubit_numpy(gate, target_qubit, num_qubits, state=None):
    """
    Apply a single-qubit gate to a specific qubit in a multi-qubit system.

    Args:
        gate: The single-qubit gate to apply
        target_qubit: The index of the target qubit (0-indexed)
        num_qubits: Total number of qubits in the system
        state: Optional quantum state to apply the gate to

    Returns:
        tuple: (Full operator, resulting state if state was provided)
    """
    # Create a list of identity operators
    operators = [I] * num_qubits

    # Replace the target qubit's operator with the gate
    operators[target_qubit] = gate

    # Calculate the tensor product of all operators
    full_operator = operators[0]
    for op in operators[1:]:
        full_operator = np.kron(full_operator, op)

    if state is not None:
        return full_operator, np.dot(full_operator, state)
    else:
        return full_operator, None

# Example: Apply Hadamard to the first qubit in a 2-qubit system |00⟩
state_00 = tensor_product_numpy([state_0, state_0])
print("Initial state |00⟩:")
print(state_00)

print("\nApplying H to the first qubit:")
_, result_H_first = apply_gate_to_qubit_numpy(H, 0, 2, state_00)
print(result_H_first)

print("\nApplying X to the second qubit:")
_, result_X_second = apply_gate_to_qubit_numpy(X, 1, 2, state_00)
print(result_X_second)
```


## SYMPY

## Part 2: Qubit Modeling with SymPy

Now let's implement the same operations using SymPy, which provides symbolic mathematics capabilities.

```python
import sympy as sp
from sympy.physics.quantum import TensorProduct
from sympy import Matrix, I as sp_I, sqrt, simplify
```

### 2.1 Checking if a Complex Vector Forms a Valid Qubit

```python
def is_valid_qubit_sympy(state_vector):
    """
    Check if a vector represents a valid qubit state using SymPy.

    Args:
        state_vector: A SymPy Matrix of shape (2, 1) containing complex numbers

    Returns:
        bool: True if the state is a valid qubit, False otherwise
    """
    # Ensure the vector has the right shape
    if state_vector.shape != (2, 1):
        return False

    # Calculate the norm and check if it's 1
    norm_squared = 0
    for element in state_vector:
        norm_squared += (element.conjugate() * element)

    return sp.simplify(norm_squared - 1) == 0

# Examples with SymPy
print("SymPy Qubit Validation Examples:")

state_0_sp = Matrix([1, 0])
print("Is |0⟩ valid:", is_valid_qubit_sympy(state_0_sp))

state_1_sp = Matrix([0, 1])
print("Is |1⟩ valid:", is_valid_qubit_sympy(state_1_sp))

state_plus_sp = Matrix([1/sqrt(2), 1/sqrt(2)])
print("Is |+⟩ valid:", is_valid_qubit_sympy(state_plus_sp))

invalid_state_sp = Matrix([1/2, 1/2])
print("Is [1/2, 1/2] valid:", is_valid_qubit_sympy(invalid_state_sp))

# Using symbolic variables
alpha = sp.symbols('alpha', complex=True)
beta = sp.symbols('beta', complex=True)
symbolic_state = Matrix([alpha, beta])
print("\nSymbolic state [alpha, beta]:")
print("Is valid if |alpha|² + |beta|² = 1:",
      sp.simplify(alpha.conjugate()*alpha + beta.conjugate()*beta) == 1)
```

### 2.2 Calculating Tensor Products of Multiple Qubits

```python
def tensor_product_sympy(qubit_list):
    """
    Calculate the tensor product of multiple qubit states using SymPy.

    Args:
        qubit_list: List of SymPy matrices, each representing a qubit state

    Returns:
        sympy.Matrix: Tensor product of all qubits
    """
    result = qubit_list[0]
    for qubit in qubit_list[1:]:
        result = TensorProduct(result, qubit)
    return result

# Examples
print("SymPy Tensor Product Examples:")
tensor_00_sp = tensor_product_sympy([state_0_sp, state_0_sp])
print("Tensor product of |0⟩ ⊗ |0⟩:")
print(tensor_00_sp)

tensor_01_sp = tensor_product_sympy([state_0_sp, state_1_sp])
print("\nTensor product of |0⟩ ⊗ |1⟩:")
print(tensor_01_sp)

tensor_plusplus_sp = tensor_product_sympy([state_plus_sp, state_plus_sp])
print("\nTensor product of |+⟩ ⊗ |+⟩:")
print(tensor_plusplus_sp)
```

### 2.3 Verifying if a Matrix is Unitary

```python
def is_unitary_sympy(matrix):
    """
    Check if a matrix is unitary using SymPy.

    Args:
        matrix: A square SymPy Matrix

    Returns:
        bool: True if the matrix is unitary, False otherwise
    """
    if matrix.shape[0] != matrix.shape[1]:
        return False

    # Calculate U†U
    conj_transpose = matrix.H
    product = conj_transpose * matrix

    # Check if U†U is the identity matrix
    identity = sp.eye(matrix.shape[0])
    return sp.simplify(product - identity) == sp.zeros(*product.shape)

# Define quantum gates using SymPy
H_sp = Matrix([[1, 1], [1, -1]]) / sqrt(2)  # Hadamard gate
X_sp = Matrix([[0, 1], [1, 0]])             # Pauli-X gate
Y_sp = Matrix([[0, -sp_I], [sp_I, 0]])      # Pauli-Y gate
Z_sp = Matrix([[1, 0], [0, -1]])            # Pauli-Z gate
I_sp = sp.eye(2)                            # Identity gate

print("Check if common quantum gates are unitary (SymPy):")
print(f"Hadamard (H): {is_unitary_sympy(H_sp)}")
print(f"Pauli-X: {is_unitary_sympy(X_sp)}")
print(f"Pauli-Y: {is_unitary_sympy(Y_sp)}")
print(f"Pauli-Z: {is_unitary_sympy(Z_sp)}")
print(f"Identity: {is_unitary_sympy(I_sp)}")

# Create a non-unitary matrix for comparison
non_unitary_sp = Matrix([[1, 1], [1, 1]])
print(f"Non-unitary matrix: {is_unitary_sympy(non_unitary_sp)}")
```

### 2.4 Calculating the Effect of a Unitary Matrix on Qubits

```python
def apply_unitary_sympy(unitary, state):
    """
    Apply a unitary operation to a quantum state using SymPy.

    Args:
        unitary: A unitary matrix as SymPy Matrix
        state: A quantum state vector as SymPy Matrix

    Returns:
        sympy.Matrix: The resulting quantum state
    """
    result = unitary * state
    return sp.simplify(result)

# Examples
print("Applying unitary operations to qubit states (



### breakpoint
# Qubit Modeling with SymPy

This notebook demonstrates how to use SymPy to model quantum bits (qubits) and perform fundamental quantum operations. We'll cover:

1. Checking if a pair of complex numbers forms a valid qubit
2. Calculating tensor products of multiple qubits
3. Verifying if a matrix is unitary
4. Calculating the effect of unitary matrices on qubit systems

## Setup and Imports

Let's begin by importing the necessary libraries:

```python
import sympy as sp
from sympy import Matrix, I, symbols, sqrt, simplify, expand, Rational
from sympy.physics.quantum import TensorProduct
import numpy as np  # For numerical comparisons
import matplotlib.pyplot as plt  # For visualization
from IPython.display import display, Math
```

## 1. Representing Qubits with SymPy

A qubit is represented as a 2-dimensional complex vector $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$ where $\alpha$ and $\beta$ are complex numbers that satisfy $|\alpha|^2 + |\beta|^2 = 1$.

### 1.1 Creating Basic Qubits

```python
# Define the standard basis states
ket_0 = Matrix([1, 0])
ket_1 = Matrix([0, 1])

# Define some common quantum states
ket_plus = Matrix([1/sqrt(2), 1/sqrt(2)])  # |+⟩ = (|0⟩ + |1⟩)/√2
ket_minus = Matrix([1/sqrt(2), -1/sqrt(2)]) # |-⟩ = (|0⟩ - |1⟩)/√2

# Display the states using LaTeX
print("Basic qubit states:")
display(Math(r'|0\rangle = ' + sp.latex(ket_0)))
display(Math(r'|1\rangle = ' + sp.latex(ket_1)))
display(Math(r'|+\rangle = ' + sp.latex(ket_plus)))
display(Math(r'|-\rangle = ' + sp.latex(ket_minus)))

# We can also create qubits using symbolic coefficients
alpha, beta = symbols('alpha beta', complex=True)
symbolic_qubit = Matrix([alpha, beta])
display(Math(r'|\psi\rangle = ' + sp.latex(symbolic_qubit)))
```

### 1.2 Checking if a Pair of Complex Numbers Forms a Valid Qubit

For a state to be a valid qubit, it must be normalized, meaning $|\alpha|^2 + |\beta|^2 = 1$.

```python
def is_valid_qubit(state):
    """
    Check if a state vector represents a valid qubit.

    Args:
        state: A SymPy Matrix of shape (2, 1) or (2,)

    Returns:
        bool: True if the state is a valid qubit, False otherwise
    """
    # Ensure proper dimensions
    if state.shape != (2, 1) and state.shape != (2,):
        return False

    # Calculate the norm
    norm_squared = 0
    for element in state:
        norm_squared += element.conjugate() * element

    # Check if the norm is 1
    simplify_norm = simplify(norm_squared)
    return simplify_norm == 1

# Test with our defined states
print("Validity check of qubit states:")
print(f"Is |0⟩ a valid qubit? {is_valid_qubit(ket_0)}")
print(f"Is |1⟩ a valid qubit? {is_valid_qubit(ket_1)}")
print(f"Is |+⟩ a valid qubit? {is_valid_qubit(ket_plus)}")
print(f"Is |-⟩ a valid qubit? {is_valid_qubit(ket_minus)}")

# Test with an invalid state
invalid_state = Matrix([1, 1])  # Not normalized
print(f"Is [1, 1] a valid qubit? {is_valid_qubit(invalid_state)}")

# For symbolic states, we need to impose the normalization condition
print("\nFor symbolic state |ψ⟩:")
norm_condition = simplify(alpha.conjugate() * alpha + beta.conjugate() * beta)
display(Math(r'|\alpha|^2 + |\beta|^2 = ' + sp.latex(norm_condition)))
print("The condition for |ψ⟩ to be a valid qubit is:")
display(Math(sp.latex(norm_condition) + r' = 1'))
```

### 1.3 Creating Arbitrary Qubit States

We can create arbitrary qubit states using angles on the Bloch sphere.

```python
def bloch_state(theta, phi):
    """
    Create a qubit state from Bloch sphere coordinates.

    Args:
        theta: Polar angle (0 to π)
        phi: Azimuthal angle (0 to 2π)

    Returns:
        sympy.Matrix: A qubit state corresponding to the given angles
    """
    return Matrix([sp.cos(theta/2), sp.exp(I*phi) * sp.sin(theta/2)])

# Create some states using Bloch sphere coordinates
theta, phi = symbols('theta phi', real=True)
bloch_qubit = bloch_state(theta, phi)

# Show parametrized state
display(Math(r'|\psi(\theta, \phi)\rangle = ' + sp.latex(bloch_qubit)))

# Specific examples
equator_state = bloch_state(sp.pi/2, sp.pi/4)
display(Math(r'|\psi(\pi/2, \pi/4)\rangle = ' + sp.latex(simplify(equator_state))))

# Verify it's a valid qubit
print(f"Is the Bloch sphere state valid? {is_valid_qubit(bloch_qubit)}")
```

## 2. Calculating Tensor Products of Multiple Qubits

The tensor product allows us to combine multiple qubits into a larger quantum system.

### 2.1 Basic Tensor Products

```python
def tensor_product(qubit_list):
    """
    Calculate the tensor product of multiple qubit states.

    Args:
        qubit_list: List of SymPy Matrices, each representing a qubit state

    Returns:
        sympy.Matrix: Combined state representing the tensor product
    """
    result = qubit_list[0]
    for qubit in qubit_list[1:]:
        result = TensorProduct(result, qubit)
    return result

# Calculate some common tensor products
print("Tensor products of qubit states:")

# |00⟩
state_00 = tensor_product([ket_0, ket_0])
display(Math(r'|0\rangle \otimes |0\rangle = |00\rangle = ' + sp.latex(state_00)))

# |01⟩
state_01 = tensor_product([ket_0, ket_1])
display(Math(r'|0\rangle \otimes |1\rangle = |01\rangle = ' + sp.latex(state_01)))

# |10⟩
state_10 = tensor_product([ket_1, ket_0])
display(Math(r'|1\rangle \otimes |0\rangle = |10\rangle = ' + sp.latex(state_10)))

# |11⟩
state_11 = tensor_product([ket_1, ket_1])
display(Math(r'|1\rangle \otimes |1\rangle = |11\rangle = ' + sp.latex(state_11)))
```

### 2.2 Creating Entangled States

Let's create some famous entangled states using the tensor product and superposition:

```python
# Create Bell states (maximally entangled two-qubit states)

# Bell state Φ+ = (|00⟩ + |11⟩)/√2
bell_phi_plus = (tensor_product([ket_0, ket_0]) + tensor_product([ket_1, ket_1])) / sqrt(2)
display(Math(r'\Phi^+ = \frac{|00\rangle + |11\rangle}{\sqrt{2}} = ' + sp.latex(bell_phi_plus)))

# Bell state Φ- = (|00⟩ - |11⟩)/√2
bell_phi_minus = (tensor_product([ket_0, ket_0]) - tensor_product([ket_1, ket_1])) / sqrt(2)
display(Math(r'\Phi^- = \frac{|00\rangle - |11\rangle}{\sqrt{2}} = ' + sp.latex(bell_phi_minus)))

# Bell state Ψ+ = (|01⟩ + |10⟩)/√2
bell_psi_plus = (tensor_product([ket_0, ket_1]) + tensor_product([ket_1, ket_0])) / sqrt(2)
display(Math(r'\Psi^+ = \frac{|01\rangle + |10\rangle}{\sqrt{2}} = ' + sp.latex(bell_psi_plus)))

# Bell state Ψ- = (|01⟩ - |10⟩)/√2
bell_psi_minus = (tensor_product([ket_0, ket_1]) - tensor_product([ket_1, ket_0])) / sqrt(2)
display(Math(r'\Psi^- = \frac{|01\rangle - |10\rangle}{\sqrt{2}} = ' + sp.latex(bell_psi_minus)))

# Verify these are valid quantum states
print("\nValidity check of Bell states:")
print(f"Is Φ+ valid? {is_valid_qubit(bell_phi_plus)}")
print(f"Is Ψ- valid? {is_valid_qubit(bell_psi_minus)}")
```

### 2.3 Three-Qubit Systems

```python
# Create a three-qubit system |000⟩
state_000 = tensor_product([ket_0, ket_0, ket_0])
display(Math(r'|000\rangle = ' + sp.latex(state_000)))

# Create the GHZ state (|000⟩ + |111⟩)/√2
ghz_state = (tensor_product([ket_0, ket_0, ket_0]) + tensor_product([ket_1, ket_1, ket_1])) / sqrt(2)
display(Math(r'GHZ = \frac{|000\rangle + |111\rangle}{\sqrt{2}} = ' + sp.latex(ghz_state)))

# Create the W state (|100⟩ + |010⟩ + |001⟩)/√3
w_state = (tensor_product([ket_1, ket_0, ket_0]) +
           tensor_product([ket_0, ket_1, ket_0]) +
           tensor_product([ket_0, ket_0, ket_1])) / sqrt(3)
display(Math(r'W = \frac{|100\rangle + |010\rangle + |001\rangle}{\sqrt{3}} = ' + sp.latex(w_state)))
```

## 3. Verifying if a Matrix is Unitary

A matrix U is unitary if U†U = I, where U† is the conjugate transpose of U and I is the identity matrix.

### 3.1 Defining Common Quantum Gates

```python
# Define common single-qubit gates
I_gate = Matrix([[1, 0], [0, 1]])            # Identity gate
X_gate = Matrix([[0, 1], [1, 0]])            # Pauli-X (NOT gate)
Y_gate = Matrix([[0, -I], [I, 0]])           # Pauli-Y
Z_gate = Matrix([[1, 0], [0, -1]])           # Pauli-Z
H_gate = Matrix([[1, 1], [1, -1]]) / sqrt(2) # Hadamard

# Show the gates in LaTeX
print("Common quantum gates:")
display(Math(r'I = ' + sp.latex(I_gate)))
display(Math(r'X = ' + sp.latex(X_gate)))
display(Math(r'Y = ' + sp.latex(Y_gate)))
display(Math(r'Z = ' + sp.latex(Z_gate)))
display(Math(r'H = ' + sp.latex(H_gate)))

# Define rotation gates
def rotation_x(theta):
    return Matrix([[sp.cos(theta/2), -I*sp.sin(theta/2)],
                   [-I*sp.sin(theta/2), sp.cos(theta/2)]])

def rotation_y(theta):
    return Matrix([[sp.cos(theta/2), -sp.sin(theta/2)],
                   [sp.sin(theta/2), sp.cos(theta/2)]])

def rotation_z(theta):
    return Matrix([[sp.exp(-I*theta/2), 0],
                  [0, sp.exp(I*theta/2)]])

# Display a rotation gate example
theta_val = symbols('theta', real=True)
display(Math(r'R_x(\theta) = ' + sp.latex(rotation_x(theta_val))))
```

### 3.2 Checking Unitarity

```python
def is_unitary(matrix):
    """
    Check if a matrix is unitary.

    Args:
        matrix: A square SymPy Matrix

    Returns:
        bool: True if the matrix is unitary, False otherwise
    """
    if matrix.shape[0] != matrix.shape[1]:
        return False

    # Calculate U†U
    conj_transpose = matrix.H
    product = conj_transpose * matrix

    # Check if U†U equals the identity matrix
    identity = sp.eye(matrix.shape[0])
    result = simplify(product - identity)

    # Check if all elements are approximately zero
    return all(abs(complex(element)) < 1e-10 for element in result)

# Test unitarity of common quantum gates
print("Unitarity check of quantum gates:")
print(f"Is I unitary? {is_unitary(I_gate)}")
print(f"Is X unitary? {is_unitary(X_gate)}")
print(f"Is Y unitary? {is_unitary(Y_gate)}")
print(f"Is Z unitary? {is_unitary(Z_gate)}")
print(f"Is H unitary? {is_unitary(H_gate)}")

# Test with a non-unitary matrix
non_unitary = Matrix([[1, 1], [1, 1]])
print(f"Is [[1, 1], [1, 1]] unitary? {is_unitary(non_unitary)}")

# Test rotation gates
print(f"Is Rx(π) unitary? {is_unitary(rotation_x(sp.pi))}")
print(f"Is Ry(π/2) unitary? {is_unitary(rotation_y(sp.pi/2))}")
```

### 3.3 Demonstrating Unitarity Properties

```python
# Show that the product of unitary matrices is unitary
product_HX = H_gate * X_gate
print(f"Is H·X unitary? {is_unitary(product_HX)}")

# Show that the tensor product of unitary matrices is unitary
tensor_HX = TensorProduct(H_gate, X_gate)
print(f"Is H⊗X unitary? {is_unitary(tensor_HX)}")

# Demonstrate symbolic unitarity check with a rotation gate
rx_theta = rotation_x(theta_val)
conj_


### BREAK

import sympy as sp
from sympy import Matrix, I, sqrt, eye, simplify, expand
from sympy.physics.quantum import TensorProduct

def print_separator():
    """Print a separator line for better readability."""
    print("\n" + "="*50 + "\n")

def is_unitary(matrix):
    """
    Check if a matrix is unitary (U†U = I).

    Args:
        matrix: A square SymPy Matrix

    Returns:
        bool: True if the matrix is unitary, False otherwise
    """
    if matrix.shape[0] != matrix.shape[1]:
        return False

    # Calculate U†U (conjugate transpose multiplied by the original matrix)
    conj_transpose = matrix.H
    product = conj_transpose * matrix

    # Check if U†U equals the identity matrix
    identity = eye(matrix.shape[0])
    result = simplify(product - identity)

    # Matrix is unitary if this difference is zero
    return result.is_zero_matrix

def apply_unitary(unitary, state):
    """
    Apply a unitary operation to a quantum state.

    Args:
        unitary: A unitary matrix as SymPy Matrix
        state: A quantum state vector as SymPy Matrix

    Returns:
        sympy.Matrix: The resulting quantum state
    """
    return simplify(unitary * state)

def tensor_product(qubit_list):
    """
    Calculate the tensor product of multiple qubit states.

    Args:
        qubit_list: List of SymPy Matrices, each representing a qubit state

    Returns:
        sympy.Matrix: Combined state representing the tensor product
    """
    result = qubit_list[0]
    for qubit in qubit_list[1:]:
        result = TensorProduct(result, qubit)
    return result

def is_valid_qubit(state):
    """
    Check if a state vector represents a valid qubit.

    Args:
        state: A SymPy Matrix of shape (2, 1) or (2,)

    Returns:
        bool: True if the state is a valid qubit, False otherwise
    """
    # Ensure proper dimensions
    if state.shape != (2, 1) and state.shape != (2,):
        return False

    # Calculate the norm squared
    norm_squared = 0
    for element in state:
        norm_squared += element.conjugate() * element

    # Check if the norm is 1
    return simplify(norm_squared - 1) == 0

def print_state(state, name):
    """Print a quantum state in a readable format."""
    print(f"{name} = {state}")
    print(f"Is valid qubit: {is_valid_qubit(state)}")

# Define basis states
print("Defining basis states:")
ket_0 = Matrix([1, 0])
ket_1 = Matrix([0, 1])
print_state(ket_0, "|0⟩")
print_state(ket_1, "|1⟩")

# Define some common quantum states
print("\nDefining common quantum states:")
ket_plus = Matrix([1/sqrt(2), 1/sqrt(2)])
ket_minus = Matrix([1/sqrt(2), -1/sqrt(2)])
print_state(ket_plus, "|+⟩")
print_state(ket_minus, "|-⟩")

print_separator()

# Define common quantum gates
print("Defining common quantum gates:")
I_gate = Matrix([[1, 0], [0, 1]])              # Identity gate
X_gate = Matrix([[0, 1], [1, 0]])              # Pauli-X (NOT gate)
Y_gate = Matrix([[0, -I], [I, 0]])             # Pauli-Y
Z_gate = Matrix([[1, 0], [0, -1]])             # Pauli-Z
H_gate = Matrix([[1, 1], [1, -1]]) / sqrt(2)   # Hadamard
S_gate = Matrix([[1, 0], [0, I]])              # Phase gate
T_gate = Matrix([[1, 0], [0, exp(I*sp.pi/4)]]) # T gate
CNOT = Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])  # CNOT gate

print("Common quantum gates defined:")
print(f"I = {I_gate}")
print(f"X = {X_gate}")
print(f"Y = {Y_gate}")
print(f"Z = {Z_gate}")
print(f"H = {H_gate}")
print(f"S = {S_gate}")
print(f"T = {T_gate}")
print(f"CNOT = {CNOT}")

print_separator()

# Verify unitarity of quantum gates
print("Verifying unitarity of quantum gates:")
print(f"Is I unitary? {is_unitary(I_gate)}")
print(f"Is X unitary? {is_unitary(X_gate)}")
print(f"Is Y unitary? {is_unitary(Y_gate)}")
print(f"Is Z unitary? {is_unitary(Z_gate)}")
print(f"Is H unitary? {is_unitary(H_gate)}")
print(f"Is S unitary? {is_unitary(S_gate)}")
print(f"Is T unitary? {is_unitary(T_gate)}")
print(f"Is CNOT unitary? {is_unitary(CNOT)}")

# Test with a non-unitary matrix
non_unitary = Matrix([[1, 1], [1, 1]])
print(f"Is [[1, 1], [1, 1]] unitary? {is_unitary(non_unitary)}")

print_separator()

# Calculate the effect of unitary matrices on qubit states
print("Calculating the effect of unitary matrices on qubit states:")

print("\nEffect of X gate (NOT gate) on |0⟩:")
result_X0 = apply_unitary(X_gate, ket_0)
print_state(result_X0, "X|0⟩")
print(f"Is X|0⟩ = |1⟩? {result_X0 == ket_1}")

print("\nEffect of X gate on |1⟩:")
result_X1 = apply_unitary(X_gate, ket_1)
print_state(result_X1, "X|1⟩")
print(f"Is X|1⟩ = |0⟩? {result_X1 == ket_0}")

print("\nEffect of H gate on |0⟩:")
result_H0 = apply_unitary(H_gate, ket_0)
print_state(result_H0, "H|0⟩")
print(f"Is H|0⟩ = |+⟩? {simplify(result_H0 - ket_plus).is_zero_matrix}")

print("\nEffect of H gate on |1⟩:")
result_H1 = apply_unitary(H_gate, ket_1)
print_state(result_H1, "H|1⟩")
print(f"Is H|1⟩ = |-⟩? {simplify(result_H1 - ket_minus).is_zero_matrix}")

print("\nEffect of Z gate on |+⟩:")
result_Zplus = apply_unitary(Z_gate, ket_plus)
print_state(result_Zplus, "Z|+⟩")
print(f"Is Z|+⟩ = |-⟩? {simplify(result_Zplus - ket_minus).is_zero_matrix}")

print_separator()

# Multi-qubit systems
print("Working with multi-qubit systems:")

# Create some multi-qubit states
print("\nCreating multi-qubit states:")
state_00 = tensor_product([ket_0, ket_0])
state_01 = tensor_product([ket_0, ket_1])
state_10 = tensor_product([ket_1, ket_0])
state_11 = tensor_product([ket_1, ket_1])

print(f"|00⟩ = {state_00}")
print(f"|01⟩ = {state_01}")
print(f"|10⟩ = {state_10}")
print(f"|11⟩ = {state_11}")

# Bell state (entangled state)
bell_state = (state_00 + state_11) / sqrt(2)
print(f"\nBell state (|00⟩ + |11⟩)/√2 = {bell_state}")
print(f"Is Bell state valid? {is_valid_qubit(bell_state)}")

print_separator()

# Apply CNOT gate to different states
print("Applying CNOT gate to different states:")
print("\nApplying CNOT to |00⟩:")
result_CNOT_00 = apply_unitary(CNOT, state_00)
print(f"CNOT|00⟩ = {result_CNOT_00}")
print(f"Is CNOT|00⟩ = |00⟩? {simplify(result_CNOT_00 - state_00).is_zero_matrix}")

print("\nApplying CNOT to |10⟩:")
result_CNOT_10 = apply_unitary(CNOT, state_10)
print(f"CNOT|10⟩ = {result_CNOT_10}")
print(f"Is CNOT|10⟩ = |11⟩? {simplify(result_CNOT_10 - state_11).is_zero_matrix}")

print("\nApplying CNOT to |+0⟩:")
state_plus0 = tensor_product([ket_plus, ket_0])
result_CNOT_plus0 = apply_unitary(CNOT, state_plus0)
print(f"|+0⟩ = {state_plus0}")
print(f"CNOT|+0⟩ = {result_CNOT_plus0}")
print(f"Is CNOT|+0⟩ = Bell state? {simplify(result_CNOT_plus0 - bell_state).is_zero_matrix}")

print_separator()

# Creating a quantum circuit (sequence of gates)
print("Creating a quantum circuit:")
print("Example: Applying H ⊗ I followed by CNOT to |00⟩")

# Create H ⊗ I
H_tensor_I = TensorProduct(H_gate, I_gate)
print(f"H ⊗ I = {H_tensor_I}")
print(f"Is H ⊗ I unitary? {is_unitary(H_tensor_I)}")

# Apply H ⊗ I to |00⟩
intermediate_state = apply_unitary(H_tensor_I, state_00)
print(f"(H ⊗ I)|00⟩ = {intermediate_state}")
print(f"Is (H ⊗ I)|00⟩ = |+0⟩? {simplify(intermediate_state - state_plus0).is_zero_matrix}")

# Apply CNOT to the intermediate state
final_state = apply_unitary(CNOT, intermediate_state)
print(f"CNOT(H ⊗ I)|00⟩ = {final_state}")
print(f"Is CNOT(H ⊗ I)|00⟩ = Bell state? {simplify(final_state - bell_state).is_zero_matrix}")

print_separator()

# Demonstrate operation on a larger system
print("Working with a three-qubit system:")

# Create a three-qubit state |000⟩
state_000 = tensor_product([ket_0, ket_0, ket_0])
print(f"|000⟩ = {state_000}")

# Apply H to the first qubit (H ⊗ I ⊗ I)
H_I_I = TensorProduct(H_gate, I_gate, I_gate)
state_plus00 = apply_unitary(H_I_I, state_000)
print(f"(H ⊗ I ⊗ I)|000⟩ = {state_plus00}")

# Create a Toffoli gate (CCNOT)
Toffoli = Matrix([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0]
])

print(f"Is Toffoli unitary? {is_unitary(Toffoli)}")

# Create a GHZ state (|000⟩ + |111⟩)/√2
# Apply H to first qubit, then CNOT to first and second, then CNOT to second and third
CNOT_12 = Matrix([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0]
])

CNOT_23 = Matrix([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0]
])

# Apply H to first qubit
state_after_H = apply_unitary(H_I_I, state_000)
print(f"(H ⊗ I ⊗ I)|000⟩ = {state_after_H}")

# Apply CNOT to first and second qubits
state_after_


### COMPARE

# NumPy vs SymPy for Modeling Qubits

NumPy and SymPy offer different approaches to modeling quantum systems like qubits, with distinct strengths and limitations:

## NumPy
NumPy excels at numerical computation with qubits:

- **Numerical focus**: Represents quantum states as numerical arrays and matrices
- **Performance**: Highly optimized for fast numerical calculations on large arrays
- **State representation**: Qubits are represented as complex-valued vectors
- **Operations**: Quantum gates implemented as matrix operations
- **Measurement**: Probabilistic outcomes simulated through numerical sampling
- **Limitations**: Lacks symbolic manipulation; numerical approximations can introduce errors

## SymPy
SymPy provides symbolic computation for quantum systems:

- **Symbolic focus**: Represents quantum states and operations symbolically
- **Exact expressions**: Maintains mathematical precision through symbolic manipulation
- **State representation**: Can represent qubits with symbolic coefficients (e.g., √(1/2)|0⟩ + √(1/2)|1⟩)
- **Operations**: Quantum transformations as symbolic matrix operations
- **Measurement**: Can derive exact probability expressions
- **Limitations**: Slower performance for large systems; not optimized for numerical simulations

## When to Use Each:

- **Use NumPy when**: You need numerical simulations of quantum systems, performance is critical, or you're working with many qubits
- **Use SymPy when**: You need exact symbolic expressions, are deriving theoretical results, or teaching quantum concepts with precise mathematical notation

For serious quantum computing applications, you might consider specialized libraries like Qiskit, Cirq, or QuTiP, which are built specifically for quantum simulation and offer more comprehensive tools.
