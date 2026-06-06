# The gate-application kernel

This note derives the arithmetic behind `src/kernel.rs`. The goal is to apply a
gate to a statevector *in place*, touching only the amplitudes the gate actually
mixes, without ever building a $2^n \times 2^n$ matrix.

## Statevector layout

The state of $n$ qubits is a vector of $N = 2^n$ complex amplitudes. The
amplitude at index $j$ is the coefficient of the computational basis state
$|j\rangle$. We use **little-endian** qubit order: qubit $k$ is bit $k$ of the
index, so

$$j = \sum_{k=0}^{n-1} b_k\, 2^k, \qquad b_k \in \{0, 1\},$$

and qubit $0$ is the least-significant bit. A single-qubit state is
$|\psi\rangle = \alpha\,|0\rangle + \beta\,|1\rangle$ with
$|\alpha|^2 + |\beta|^2 = 1$.

## Single-qubit gates: the bit-insertion loop

A single-qubit gate $U$ with entries $u_{00}, u_{01}, u_{10}, u_{11}$ on
qubit $k$ acts independently on each pair of basis states that differ only in
bit $k$:

$$\begin{pmatrix} \psi'_{i_0} \\ \psi'_{i_1} \end{pmatrix} = \begin{pmatrix} u_{00} & u_{01} \\ u_{10} & u_{11} \end{pmatrix} \begin{pmatrix} \psi_{i_0} \\ \psi_{i_1} \end{pmatrix},$$

where $i_0$ has bit $k$ clear and $i_1 = i_0 \mathbin{|} 2^k$ has it set. There
are exactly $N/2 = 2^{n-1}$ such pairs.

To enumerate them, let a counter $i$ run over $0 \le i < 2^{n-1}$ and *insert a
zero bit at position $k$*. The low $k$ bits of $i$ pass through unchanged; the
remaining bits shift up by one to vacate position $k$:

$$i_0 = \bigl((i \gg k) \ll (k+1)\bigr) \;\mathbin{|}\; \bigl(i \wedge (2^k - 1)\bigr), \qquad i_1 = i_0 \mathbin{|} 2^k .$$

**Why this is correct.** Write $i = h \cdot 2^k + \ell$ with
$\ell = i \bmod 2^k$ the low part and $h = i \gg k$ the high part. Then
$i_0 = h \cdot 2^{k+1} + \ell$, which has bit $k$ equal to $0$ by construction.
The map $i \mapsto i_0$ is a bijection from $\{0, \dots, 2^{n-1}-1\}$ onto the
$2^{n-1}$ indices with bit $k$ clear, and each such $i_0$ pairs with a distinct
$i_1$. So across the loop **every index in $\{0, \dots, N-1\}$ is written exactly
once**: the loop is a permutation of the amplitude array. This is the property
the Kani proofs in `kernel::proofs` establish for the unsafe fast path:

- $i_0 < N$ and $i_1 < N$ (in bounds),
- $i_0 \neq i_1$ (a real pair),
- $i \neq j \Rightarrow i_0(i) \neq i_0(j)$ (injective, hence a bijection).

Each pair costs four complex multiplications and two additions; the whole gate
is $O(N)$ with no allocation.

## Two-qubit gates: groups of four

A gate on qubits $a$ and $b$ mixes the four basis states that differ only in
those two bits. We insert *two* zero bits (at the lower target position first,
then the higher) to build a base index with both target bits clear, then set
the bits to enumerate the group:

$$\{\,i_{00},\; i_{01},\; i_{10},\; i_{11}\,\} = \{\, \text{base},\; \text{base} \mathbin{|} 2^b,\; \text{base} \mathbin{|} 2^a,\; \text{base} \mathbin{|} 2^a \mathbin{|} 2^b \,\}.$$

There are $N/4 = 2^{n-2}$ groups, and the gate applies its $4 \times 4$ matrix to
each. Several common gates collapse to something cheaper than a full matrix
multiply:

- **CNOT** swaps $\psi_{i_{10}} \leftrightarrow \psi_{i_{11}}$ (flip the target
  when the control is set).
- **CZ** negates $\psi_{i_{11}}$ only.
- **SWAP** exchanges $\psi_{i_{01}} \leftrightarrow \psi_{i_{10}}$.

`everett` dispatches CNOT, CZ, and SWAP to these specialized loops (no complex
multiply) and applies the dense $4 \times 4$ matrix for every other two-qubit
gate.

## Controlled gates

A gate controlled on a set $C$ of qubits applies its single-qubit core only on
the subspace where every control bit is set. Using the single-qubit pairing
above, build the control mask $m = \sum_{c \in C} 2^c$ and apply the $2 \times 2$
update to the pair $(i_0, i_1)$ only when $i_0 \wedge m = m$. Because $i_0$
and $i_1$ agree on every bit except the target, testing $i_0$ alone is
sufficient.

## Tensor-contraction view

The same operation is a tensor contraction. Reshape the statevector into a
rank-$n$ tensor $T_{b_{n-1} \cdots b_1 b_0}$ with one index per qubit. A gate on
qubit $k$ contracts its matrix against index $b_k$:

$$T'_{\cdots b_k \cdots} = \sum_{b_k'} U_{b_k\, b_k'}\, T_{\cdots b_k' \cdots},$$

holding all other indices fixed. "Iterate over the pairs that differ only in bit
$k$" is exactly "sum over the contracted index for every setting of the
others": the pair-iteration loop *is* the contraction.

## Precision

Amplitudes are `f64`. Unitary evolution preserves the norm in exact arithmetic,
but rounding causes slow drift over deep circuits; `State::normalize` rescales to
unit norm and should be called periodically and after any measurement collapse.
State comparisons use fidelity $|\langle\phi|\psi\rangle|^2$, which is invariant
under global phase: the right notion of "same state" physically.
