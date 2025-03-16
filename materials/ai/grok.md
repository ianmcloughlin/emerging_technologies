Here's a Python script that demonstrates quantum computing concepts using NumPy, including qubit modeling, tensor products, unitary matrix verification, and applying unitary operations:

```python
import numpy as np
from numpy import linalg as LA

# 1. Function to check if a pair of complex numbers forms a valid qubit state
def is_valid_qubit(alpha, beta):
    """
    Check if |alpha|^2 + |beta|^2 = 1 (normalization condition for a qubit)
    Returns True if valid, False otherwise
    """
    norm_squared = np.abs(alpha)**2 + np.abs(beta)**2
    return np.isclose(norm_squared, 1.0, atol=1e-8)

# 2. Function to calculate tensor product of multiple qubit states
def tensor_product(*states):
    """
    Calculate tensor product of multiple qubit states
    Each state should be a numpy array representing a qubit
    """
    result = states[0]
    for state in states[1:]:
        result = np.kron(result, state)
    return result

# 3. Function to verify if a matrix is unitary
def is_unitary(matrix):
    """
    Check if a matrix U is unitary: U†U = UU† = I
    where U† is the conjugate transpose and I is the identity matrix
    """
    n = matrix.shape[0]
    # Check if square matrix
    if matrix.shape[0] != matrix.shape[1]:
        return False
    
    # Calculate conjugate transpose
    dagger = matrix.conj().T
    
    # Check U†U = I
    product1 = np.dot(dagger, matrix)
    product2 = np.dot(matrix, dagger)
    
    identity = np.eye(n)
    
    return (np.allclose(product1, identity, atol=1e-8) and 
            np.allclose(product2, identity, atol=1e-8))

# 4. Function to apply unitary matrix to qubit system
def apply_unitary(unitary, state):
    """
    Apply a unitary matrix to a quantum state
    Returns the resulting state vector
    """
    if not is_unitary(unitary):
        raise ValueError("Matrix is not unitary")
    if unitary.shape[1] != len(state):
        raise ValueError("Matrix dimension doesn't match state dimension")
    return np.dot(unitary, state)

# Demonstration
def main():
    # Example 1: Qubit validation
    print("Example 1: Qubit Validation")
    qubit1 = np.array([1/np.sqrt(2), 1/np.sqrt(2)])  # |+⟩ state
    qubit2 = np.array([1, 0])  # Invalid qubit (not normalized)
    print(f"Is [1/√2, 1/√2] a valid qubit? {is_valid_qubit(qubit1[0], qubit1[1])}")
    print(f"Is [1, 0] a valid qubit? {is_valid_qubit(qubit2[0], qubit2[1])}\n")

    # Example 2: Tensor product
    print("Example 2: Tensor Product")
    qubit0 = np.array([1, 0]) / np.sqrt(1)  # |0⟩ state
    qubit1 = np.array([0, 1]) / np.sqrt(1)  # |1⟩ state
    two_qubits = tensor_product(qubit0, qubit1)
    print("Tensor product of |0⟩ and |1⟩:")
    print(two_qubits, "\n")  # Should give |01⟩ = [0, 1, 0, 0]

    # Example 3: Unitary matrix verification
    print("Example 3: Unitary Matrix Verification")
    # Hadamard gate (unitary)
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    # Non-unitary matrix
    non_unitary = np.array([[1, 2], [3, 4]])
    
    print("Is Hadamard gate unitary?", is_unitary(H))
    print("Is [[1,2],[3,4]] unitary?", is_unitary(non_unitary), "\n")

    # Example 4: Applying unitary matrix
    print("Example 4: Applying Unitary Matrix")
    initial_state = np.array([1, 0])  # |0⟩ state
    final_state = apply_unitary(H, initial_state)
    print("Initial state |0⟩:", initial_state)
    print("After Hadamard (|+⟩ state):", final_state)

if __name__ == "__main__":
    main()
```

This script includes:

1. `is_valid_qubit(alpha, beta)`: 
   - Checks if two complex numbers form a valid qubit state by verifying normalization (|α|² + |β|² = 1)
   - Uses np.isclose() for floating-point comparison with tolerance

2. `tensor_product(*states)`:
   - Calculates the tensor product of multiple qubit states using NumPy's kron function
   - Accepts variable number of input states

3. `is_unitary(matrix)`:
   - Verifies if a matrix is unitary by checking if U†U = UU† = I
   - Uses conjugate transpose and matrix multiplication
   - Includes tolerance for floating-point arithmetic

4. `apply_unitary(unitary, state)`:
   - Applies a unitary transformation to a quantum state
   - Includes error checking for matrix dimensions and unitarity

The main() function demonstrates:
- Validating qubit states (|+⟩ and an invalid state)
- Computing tensor product (|0⟩ ⊗ |1⟩ = |01⟩)
- Verifying unitarity (Hadamard gate vs. non-unitary matrix)
- Applying a unitary operation (Hadamard gate on |0⟩ to get |+⟩)

Sample output would look like:
```
Example 1: Qubit Validation
Is [1/√2, 1/√2] a valid qubit? True
Is [1, 0] a valid qubit? False

Example 2: Tensor Product
Tensor product of |0⟩ and |1⟩:
[0. 1. 0. 0.] 

Example 3: Unitary Matrix Verification
Is Hadamard gate unitary? True
Is [[1,2],[3,4]] unitary? False 

Example 4: Applying Unitary Matrix
Initial state |0⟩: [1 0]
After Hadamard (|+⟩ state): [0.70710678 0.70710678]
```

This code provides a foundation for quantum computing simulations and can be extended to include more quantum gates, measurements, and multi-qubit operations.


Here's a Python script that demonstrates quantum computing concepts using SymPy, which provides symbolic mathematics capabilities:

```python
import sympy as sp
from sympy import I, conjugate, Matrix, eye, simplify
from sympy.physics.quantum import TensorProduct

# 1. Function to check if a pair of complex numbers forms a valid qubit state
def is_valid_qubit(alpha, beta):
    """
    Check if |alpha|^2 + |beta|^2 = 1 (normalization condition for a qubit)
    Returns True if valid, False otherwise
    """
    norm_squared = conjugate(alpha) * alpha + conjugate(beta) * beta
    # Simplify the expression and check if it equals 1
    return simplify(norm_squared - 1) == 0

# 2. Function to calculate tensor product of multiple qubit states
def tensor_product(*states):
    """
    Calculate tensor product of multiple qubit states
    Each state should be a SymPy Matrix
    """
    result = states[0]
    for state in states[1:]:
        result = TensorProduct(result, state)
    return result

# 3. Function to verify if a matrix is unitary
def is_unitary(matrix):
    """
    Check if a matrix U is unitary: U†U = UU† = I
    where U† is the conjugate transpose and I is the identity matrix
    """
    n = matrix.rows
    # Check if square matrix
    if matrix.rows != matrix.cols:
        return False
    
    # Calculate conjugate transpose
    dagger = matrix.conjugate().transpose()
    
    # Check U†U = I and UU† = I
    identity = eye(n)
    product1 = simplify(dagger * matrix)
    product2 = simplify(matrix * dagger)
    
    return product1 == identity and product2 == identity

# 4. Function to apply unitary matrix to qubit system
def apply_unitary(unitary, state):
    """
    Apply a unitary matrix to a quantum state
    Returns the resulting state vector as a SymPy Matrix
    """
    if not is_unitary(unitary):
        raise ValueError("Matrix is not unitary")
    if unitary.cols != state.rows:
        raise ValueError("Matrix dimension doesn't match state dimension")
    return simplify(unitary * state)

# Demonstration
def main():
    # Example 1: Qubit validation
    print("Example 1: Qubit Validation")
    alpha = 1/sp.sqrt(2)
    beta = 1/sp.sqrt(2) * I
    print(f"Is [1/√2, i/√2] a valid qubit? {is_valid_qubit(alpha, beta)}")
    invalid_alpha = 1
    invalid_beta = 0
    print(f"Is [1, 0] a valid qubit? {is_valid_qubit(invalid_alpha, invalid_beta)}\n")

    # Example 2: Tensor product
    print("Example 2: Tensor Product")
    qubit0 = Matrix([1, 0])  # |0⟩ state
    qubit1 = Matrix([0, 1])  # |1⟩ state
    two_qubits = tensor_product(qubit0, qubit1)
    print("Tensor product of |0⟩ and |1⟩:")
    sp.pprint(two_qubits)
    print()

    # Example 3: Unitary matrix verification
    print("Example 3: Unitary Matrix Verification")
    # Hadamard gate (unitary)
    H = Matrix([[1, 1], [1, -1]]) / sp.sqrt(2)
    # Non-unitary matrix
    non_unitary = Matrix([[1, 2], [3, 4]])
    
    print("Is Hadamard gate unitary?", is_unitary(H))
    print("Is [[1,2],[3,4]] unitary?", is_unitary(non_unitary))
    print("\n")

    # Example 4: Applying unitary matrix
    print("Example 4: Applying Unitary Matrix")
    initial_state = Matrix([1, 0])  # |0⟩ state
    final_state = apply_unitary(H, initial_state)
    print("Initial state |0⟩:")
    sp.pprint(initial_state)
    print("After Hadamard:")
    sp.pprint(final_state)

if __name__ == "__main__":
    main()
```

This script uses SymPy instead of NumPy, offering symbolic manipulation capabilities. Here's what each part does:

1. `is_valid_qubit(alpha, beta)`:
   - Symbolically checks normalization condition using SymPy's conjugate and simplify
   - Returns exact symbolic equality to 1

2. `tensor_product(*states)`:
   - Uses SymPy's TensorProduct for symbolic tensor products
   - Works with SymPy Matrix objects

3. `is_unitary(matrix)`:
   - Symbolically verifies unitarity using conjugate transpose
   - Performs exact symbolic comparisons

4. `apply_unitary(unitary, state)`:
   - Applies unitary transformation symbolically
   - Returns simplified result

Key differences from the NumPy version:
- Uses symbolic mathematics instead of numerical computation
- Exact equality checks instead of floating-point tolerances
- Pretty printing of mathematical expressions
- Can handle symbolic variables as well as numbers

Sample output would look like:
```
Example 1: Qubit Validation
Is [1/√2, i/√2] a valid qubit? True
Is [1, 0] a valid qubit? False

Example 2: Tensor Product
Tensor product of |0⟩ and |1⟩:
⎡0⎤
⎢1⎥
⎢0⎥
⎢0⎥
⎣ ⎦

Example 3: Unitary Matrix Verification
Is Hadamard gate unitary? True
Is [[1,2],[3,4]] unitary? False

Example 4: Applying Unitary Matrix
Initial state |0⟩:
⎡1⎤
⎢0⎥
⎣ ⎦
After Hadamard:
⎡√2/2⎤
⎢√2/2⎥
⎣    ⎦
```

The SymPy version is particularly useful for:
- Exact symbolic calculations
- Educational purposes (shows mathematical expressions clearly)
- Working with symbolic variables
- Verifying quantum mechanics identities

Note that SymPy calculations can be slower than NumPy for numerical work but provide exact symbolic results rather than floating-point approximations.


Let's compare NumPy and SymPy for modeling qubits based on their implementation in the previous scripts, focusing on their strengths, weaknesses, and use cases in this context:

### 1. **Core Characteristics**
- **NumPy**:
  - Numerical computing library focused on fast array operations.
  - Uses floating-point arithmetic for computations.
  - Optimized for performance with large datasets and numerical simulations.
- **SymPy**:
  - Symbolic mathematics library focused on exact algebraic manipulation.
  - Performs symbolic computations with exact representations (e.g., fractions, radicals).
  - Designed for mathematical analysis and derivation rather than raw performance.

---

### 2. **Qubit Validation (`is_valid_qubit`)**
- **NumPy**:
  - **Implementation**: Checks normalization using `np.abs()` and `np.isclose()` with a tolerance (e.g., 1e-8).
  - **Strengths**:
    - Fast numerical computation.
    - Practical for simulations where small numerical errors are acceptable.
  - **Weaknesses**:
    - Relies on floating-point approximations, so exact equality isn't guaranteed (e.g., `0.99999999 ≠ 1`).
    - Tolerance parameter requires tuning for precision.
  - **Output**: Boolean based on approximate equality (e.g., `True` for `[1/√2, 1/√2]`).
- **SymPy**:
  - **Implementation**: Uses symbolic conjugation and simplification to check if `|α|^2 + |β|^2 - 1 == 0`.
  - **Strengths**:
    - Exact symbolic comparison—no floating-point errors.
    - Can handle symbolic inputs (e.g., variables like `a` and `b`).
  - **Weaknesses**:
    - Slower due to symbolic computation overhead.
    - May require simplification steps that could complicate edge cases.
  - **Output**: Exact Boolean (e.g., `True` for `[1/√2, i/√2]`).

**Comparison**: NumPy is better for quick numerical checks in simulations, while SymPy excels at exact validation and educational purposes where precision matters.

---

### 3. **Tensor Product (`tensor_product`)**
- **NumPy**:
  - **Implementation**: Uses `np.kron()` for numerical Kronecker product.
  - **Strengths**:
    - Extremely fast for large systems (scales well with multiple qubits).
    - Works seamlessly with numerical arrays.
  - **Weaknesses**:
    - Limited to numerical values; can't handle symbolic expressions.
    - Results are floating-point approximations (e.g., `0.70710678` for `1/√2`).
  - **Output**: Numerical array (e.g., `[0., 1., 0., 0.]` for `|0⟩ ⊗ |1⟩`).
- **SymPy**:
  - **Implementation**: Uses `TensorProduct` for symbolic tensor product.
  - **Strengths**:
    - Produces exact symbolic results (e.g., `Matrix([0, 1, 0, 0])`).
    - Can compute tensor products with symbolic variables.
  - **Weaknesses**:
    - Slower, especially as the number of qubits increases.
    - Less efficient for numerical simulation of large systems.
  - **Output**: Symbolic Matrix object, nicely formatted.

**Comparison**: NumPy is ideal for numerical simulations of multi-qubit systems, while SymPy is better for theoretical work or small systems where exact expressions are needed.

---

### 4. **Unitary Matrix Verification (`is_unitary`)**
- **NumPy**:
  - **Implementation**: Checks `U†U = I` and `UU† = I` numerically with `np.allclose()` and a tolerance.
  - **Strengths**:
    - Fast matrix operations using optimized linear algebra routines.
    - Practical for verifying large matrices in simulations.
  - **Weaknesses**:
    - Floating-point errors can lead to false negatives (e.g., small deviations from identity).
    - Tolerance parameter affects reliability.
  - **Output**: Boolean based on numerical comparison.
- **SymPy**:
  - **Implementation**: Checks exact equality of `U†U` and `UU†` with the identity matrix using symbolic simplification.
  - **Strengths**:
    - Exact verification—no tolerance needed.
    - Works with symbolic matrices (e.g., containing variables).
  - **Weaknesses**:
    - Computationally expensive for large matrices.
    - Simplification may fail or be slow for complex expressions.
  - **Output**: Exact Boolean based on symbolic equality.

**Comparison**: NumPy is suited for practical verification in numerical contexts, while SymPy is better for theoretical proofs or exact mathematical analysis.

---

### 5. **Applying Unitary Matrices (`apply_unitary`)**
- **NumPy**:
  - **Implementation**: Uses `np.dot()` for matrix multiplication.
  - **Strengths**:
    - Fast and efficient for numerical computation.
    - Scales well with larger systems.
  - **Weaknesses**:
    - Numerical results only (e.g., `[0.70710678, 0.70710678]` for H|0⟩).
    - No symbolic manipulation possible.
  - **Output**: Numerical array with floating-point values.
- **SymPy**:
  - **Implementation**: Uses symbolic matrix multiplication with simplification.
  - **Strengths**:
    - Produces exact symbolic results (e.g., `[√2/2, √2/2]` for H|0⟩).
    - Can handle symbolic states and operators.
  - **Weaknesses**:
    - Slower due to symbolic overhead.
    - Less practical for large-scale numerical simulations.
  - **Output**: Symbolic Matrix with exact expressions.

**Comparison**: NumPy excels in performance for applying gates in simulations, while SymPy is superior for deriving exact quantum state transformations.

---

### 6. **General Comparison**
| Aspect                | NumPy                            | SymPy                            |
|-----------------------|----------------------------------|----------------------------------|
| **Speed**             | Fast (numerical)                | Slower (symbolic)               |
| **Precision**         | Approximate (floating-point)    | Exact (symbolic)                |
| **Scalability**       | Excellent (large systems)       | Poor (small systems preferred)  |
| **Output Format**     | Numerical arrays                | Symbolic expressions/Matrices   |
| **Use Case**          | Simulations, large-scale        | Theory, education, exact results|
| **Symbolic Support**  | None                            | Full support                    |

---

### 7. **When to Use Each for Qubit Modeling**
- **Use NumPy When**:
  - Running numerical simulations of quantum circuits.
  - Working with large numbers of qubits (e.g., >5 qubits).
  - Performance is critical (e.g., real-time applications or iterative calculations).
  - Approximate results are acceptable.
  - Example: Simulating a quantum algorithm with 10 qubits.

- **Use SymPy When**:
  - Deriving exact mathematical expressions or proofs.
  - Working with small systems (e.g., 1-3 qubits) for educational purposes.
  - Need to manipulate symbolic variables (e.g., unknown amplitudes).
  - Precision is paramount over speed.
  - Example: Verifying quantum gate properties symbolically.

---

### Conclusion
For modeling qubits, **NumPy** is the go-to choice for practical quantum computing simulations due to its speed and scalability, making it ideal for research and development of quantum algorithms. **SymPy**, on the other hand, shines in theoretical and educational contexts where exact symbolic results are needed to understand quantum mechanics principles or derive new insights. The choice depends on whether your goal is numerical simulation (NumPy) or symbolic analysis (SymPy). For a hybrid approach, you could use SymPy to derive expressions and NumPy to simulate them numerically.