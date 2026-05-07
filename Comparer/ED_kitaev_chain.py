import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.sparse.linalg import expm
import sys
np.set_printoptions(threshold=sys.maxsize) ### Print all the Matrix
from typing import Optional, Tuple
from functools import reduce, partial

### Util function to construct tensor product of list of local operators
def full_tensor_product(local_operators: list) -> np.ndarray:
    """
    Iteratively computes the full tensor product for a list of local operators. 

    Parameters
    ----------
    local_operators : list
        List of local operators for which to perform the tensor product.

    Returns
    -------
    np.ndarray
        Scipy sparse array with the constructed tensor.
    """
    return reduce(sp.sparse.kron, local_operators)

### Util functions to get local operators in Fock space
def insert_operator_at_i(operator: sp.sparse.sparray, 
                         i: int, 
                         L: int, 
                         parity_sign: int = 1, 
                         dtype: type = int
                         ) -> np.ndarray:
    """
    Computes the representation of a local operator inserted at some site i in a d**L dimensional Fock space.
    This function assumes a basis such as (|down,up>, |down>, |up>, |0>) or (|0>, |down>, |up>, |down,up>)
    with parity string (1, -1, -1, 1).

    Parameters
    ----------
    operator : sp.sparse.sparray of shape (d, d)
        Local operator to be inserted.
    i : int
        Site index where the operator should be inserted.
    L : int
        Number of sites in the system. 
    parity_sign : int
        Parity sign for the repsective basis. Spin basis is +1, while charge basis uses -1.
    dtype : type
        Data type operators.

    Returns
    -------
    np.ndarray
        Representation of the operator in the d**L dimensional Fock space.
    """
    full_operators = []
    parity = sp.sparse.coo_array(parity_sign * np.array([[1, 0, 0, 0], 
                                    [0, -1, 0, 0], 
                                    [0, 0, -1, 0], 
                                    [0, 0, 0, 1]], dtype = dtype))
    identity = sp.sparse.eye_array(4, format = "coo")
    for j in range(i):
        full_operators.append(parity)
    full_operators.append(operator.tocoo())
    # full_operators.append(sp.sparse.csc_array(operator))
    for j in range(i+1, L):
        full_operators.append(identity)
    return full_tensor_product(full_operators).tocsc()  

def insert_two_operators_at_i_j(operator_i: sp.sparse.sparray, 
                                i: int, 
                                operator_j: sp.sparse.sparray, 
                                j: int, 
                                L: int, 
                                parity_sign: int = 1,
                                dtype: type = int) -> sp.sparse.sparray:
    """
    Inserts two local operators at the sites i and j, respectively.

    Parameters
    ----------
    operator_i : sp.sparse.sparray
        Local operator to be inserted at site i.
    i : int
        Site index where the first operator should be inserted.
    operator_j : sp.sparse.sparray
        Local operator to be inserted at site j.
    j : int
        Site index where the second operator should be inserted.
    L : int
        Number of sites in the system. 
    parity_sign : int
        Parity sign for the repsective basis. Spin basis is +1, while charge basis uses -1.
    dtype : type
        Data type operators.

    Returns
    -------
    np.ndarray
        Representation of the operator product in the d**L dimensional Fock space.
    """
    full_op_i = insert_operator_at_i(operator_i, i, L, parity_sign = parity_sign, dtype = dtype)
    full_op_j = insert_operator_at_i(operator_j, j, L, parity_sign = parity_sign, dtype = dtype)
    return full_op_i @ full_op_j

def insert_bosonic_operator_at_i(operator: sp.sparse.sparray, 
                                 i: int, 
                                 L: int) -> np.ndarray:
    """
    Inserts a bosonic operator at site i, i.e. an operator where you do not have to worry about the parity string.
    
    Parameters
    ----------
    operator : np.ndarray of shape (d, d)
        Local operator to be inserted.
    i : int
        Site index where the operator should be inserted.
    L : int
        Number of sites in the system. 

    Returns
    -------
    np.ndarray
        Representation of the operator in the d**L dimensional Fock space.
    """
    full_operators = [sp.sparse.eye(4, format = "csc")]*L
    full_operators[i] = operator
    return full_tensor_product(full_operators)


class ED_KitaevChain():
    def __init__(self, 
                 L: int, 
                 t: float, 
                 Delta: float, 
                 U: float, 
                 mu: float, 
                 beta: float, 
                 periodic_bc: bool = True, 
                 c_A: Optional[sp.sparse.sparray] = None, 
                 c_B: Optional[sp.sparse.sparray] = None):
        self.L = L
        self.t, self.Delta = t, Delta
        self.U, self.mu, self.beta = U, mu, beta
        self.periodic_bc = periodic_bc

        if c_A is None:
            c_up = np.array([[0, 0, 0, 0], 
                             [0, 0, 0, 0], 
                             [1, 0, 0, 0], 
                             [0, 1, 0, 0]], dtype = int)
            self.c_A = sp.sparse.csc_array(c_up)
        else:
            self.c_A = c_A
        self.c_A_dagger = self.c_A.conj().T

        if c_A is None:
            c_down = np.array([[0, 0, 0, 0], 
                               [-1, 0, 0, 0], 
                               [0, 0, 0, 0], 
                               [0, 0, 1, 0]], dtype = int)
            self.c_B = sp.sparse.csc_array(c_down)
        else:
            self.c_B = c_B
        self.c_B_dagger = self.c_B.conj().T
        self.n_A = self.c_A_dagger @ self.c_A
        self.n_B = self.c_B_dagger @ self.c_B
        self.nn = self.n_A @ self.n_B
        self.npn = self.n_A + self.n_B

        self.H_dtype = float
        self.dtype = float
        self.H = sp.sparse.csc_array((4**self.L, 4**self.L), dtype = self.H_dtype)
        self.construct_Hamiltonian()

    def construct_Hamiltonian(self):
        # Shifted chemical potential due to interaction
        mu_shift = -(self.mu + 0.5 * self.U) # minus is convention

        eff_L = self.L if self.periodic_bc else self.L-1

        ### hopping and pairing
        for i in range(eff_L):
            tmp = - self.t * insert_two_operators_at_i_j(self.c_A_dagger, i, self.c_A, (i+1)%self.L, self.L)
            self.H += tmp + tmp.conj().T
            tmp = - self.t * insert_two_operators_at_i_j(self.c_B_dagger, i, self.c_B, (i+1)%self.L, self.L)
            self.H += tmp + tmp.conj().T

            tmp = - self.Delta * insert_two_operators_at_i_j(self.c_A_dagger, i, self.c_A_dagger, (i+1)%self.L, self.L)
            self.H += tmp + tmp.conj().T
            tmp = - self.Delta * insert_two_operators_at_i_j(self.c_B_dagger, i, self.c_B_dagger, (i+1)%self.L, self.L)
            self.H += tmp + tmp.conj().T

        ### on-site interaction and chemical potential
        for i in range(self.L):
            self.H += insert_bosonic_operator_at_i(self.U*self.nn, i, self.L)
            self.H += insert_bosonic_operator_at_i(mu_shift*self.npn, i, self.L)

        self.U_op = expm(- self.beta * self.H)
        self.Z = self.U_op.trace()

    def single_particle_correlators(self, species = "A", n = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the spatial two-point correlation functions.

        Parameters
        ----------
        species : str
            Which fermion species to consider, either "A" or "B".
        n : int
            Number of evenly spaced points on which the correlation function is evaluated. 
            The total amount of points will end up being n+2 because the function will always be evaluated at tau = 0 and tau = beta
        
        Returns
        -------
        np.ndarray of shape (n+2)
            Array of sampling times (0,beta/n,...,beta)
        np.ndarray of shape (Lx,Ly,Lx,Ly,n+2)
            Array of all spatial correlators. 
        """
        assert species in ["A", "B"], "There are only the species A and B"
        c = self.c_A if species=="A" else self.c_B
        c_dagger = self.c_A_dagger if species=="A" else self.c_B_dagger
        self.t_disc = np.zeros(n + 2)
        Delta = np.round(self.beta/n, 6)
        steps = np.arange(Delta, self.beta/2 + Delta, Delta)
        self.corr = np.zeros((self.L, self.L, n+2), dtype = self.dtype)
        self.t_disc[-1] = self.beta

        ### equal time correlators
        for i1 in range(self.L):
            for i2 in range(self.L):
                tmp_c = insert_operator_at_i(c, i1, self.L, dtype = self.dtype)
                tmp_c_dagger = tmp_c.conj().T if i1==i2 else insert_operator_at_i(c_dagger, i2, self.L, dtype = self.dtype)
                self.corr[i1, i2, 0] =  (tmp_c @ tmp_c_dagger @ self.U_op).trace()
                self.corr[i1, i2, -1] =  (tmp_c @ self.U_op @ tmp_c_dagger).trace()

        ### unequal time correlators
        for k, step in enumerate(steps):
            U1 = expm( - step * self.H)
            U2 = expm( - np.round(self.beta - step, 6) * self.H)
            self.t_disc[k+1] = step
            self.t_disc[-k-2] = np.round(self.beta - step, 6)
            for i1 in range(self.L):
                for i2 in range(self.L):
                    tmp_c = insert_operator_at_i(c, i1, self.L, dtype = self.dtype)
                    tmp_c_dagger = tmp_c.conj().T if i1==i2 else insert_operator_at_i(c_dagger, i2, self.L, dtype = self.dtype)
                    self.corr[i1, i2, k+1] =  (tmp_c @ U1 @ tmp_c_dagger @ U2).trace()
                    self.corr[i1, i2, -k-2] =  (tmp_c @ U2 @ tmp_c_dagger @ U1).trace()
  
        self.corr = np.divide(self.corr, self.Z)
        return self.t_disc, self.corr

    def density_density_correlators(self, connected: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the (connected) density-density correlation function.

        Parameters
        ----------
        connected : bool
            Whether to compute the connected correlation function. 

        Returns
        -------
        np.ndarray of shape (n+2)
            Array of sampling times (0,beta/n,...,beta)
        np.ndarray of shape (Lx,Ly,Lx,Ly,n+2)
            Array of all spatial correlators. 
        """

        if self.periodic_bc:
            self.n_bar = (insert_bosonic_operator_at_i(self.npn, 0, self.L) @ self.U_op).trace() / self.Z 
            if connected:
                tmp_n_op = self.npn - self.n_bar * sp.sparse.eye(4, format = "csc")
            else:
                tmp_n_op = self.npn

            self.dd_corr = np.zeros((self.L, self.L), dtype = self.dtype)
            for i1 in range(self.L):
                n_1 = insert_bosonic_operator_at_i(tmp_n_op, i1, self.L)
                for i2 in range(self.L):
                    n_2 = n_1 if i1==i2 else insert_bosonic_operator_at_i(tmp_n_op, i2, self.L)
                    self.dd_corr[i1, i2] =  (n_1 @ n_2 @ self.U_op).trace()
        else:
            self.n_bar = np.zeros(self.L)
            for i in range(self.L):
                self.n_bar[i] = (insert_bosonic_operator_at_i(self.npn, i, self.L) @ self.U_op).trace() / self.Z

            self.dd_corr = np.zeros((self.L, self.L), dtype = self.dtype)
            for i1 in range(self.L):
                tmp_n_op = self.npn - self.n_bar[i1] * sp.sparse.eye(4, format = "csc") if connected else self.npn
                n_1 = insert_bosonic_operator_at_i(tmp_n_op, i1, self.L)
                for i2 in range(self.L):
                    tmp_n_op = self.npn - self.n_bar[i2] * sp.sparse.eye(4, format = "csc") if connected else self.npn
                    n_2 = n_1 if i1==i2 else insert_bosonic_operator_at_i(tmp_n_op, i2, self.L)
                    self.dd_corr[i1, i2] =  (n_1 @ n_2 @ self.U_op).trace()

        self.dd_corr = np.divide(self.dd_corr, self.Z)
        return self.dd_corr, self.n_bar


if __name__ == "__main__":
    L = 4
    periodic_bc = False
    t = 1
    Delta = 0.2
    beta = 0.8
    U = 1
    mu = 0

    plot_results = True

    ed_model = ED_KitaevChain(L = L, 
                              t = t, 
                              Delta = Delta, 
                              U = U, 
                              mu = mu, 
                              beta = beta, 
                              periodic_bc = periodic_bc)
    n, C = ed_model.single_particle_correlators()
    dd_corr, n_bar = ed_model.density_density_correlators()

    print("Number density: ")
    print(n_bar)

    if plot_results:
        fig, ax = plt.subplots(1, L, figsize = (L*4.8, 4.8))
        fig.supxlabel("j")
        ax[0].set_ylabel(rf"$\langle n_i n_j \rangle$")
        for x in range(L):
            cax = ax[x]
            cax.grid()
            cax.set_title(f"i={x}")
            cax.plot(dd_corr[x], linestyle = "--", label = "ED")

        plt.savefig("KitaevChain_dd_corr_ED.png", dpi = 200, bbox_inches = "tight")
        plt.close()

        fig, ax = plt.subplots(L, L, figsize = (L*4.8, L*4.8))
        fig.supylabel(r"$C_{ij}(\tau)$")
        fig.supxlabel(r"$\tau$")
        for x in range(L):
            for y in range(L):
                cax = ax[x, y]
                cax.grid()
                cax.plot(n, C[x,y,:], linestyle = "--", label = "ED")
                cax.set_title(f"(i,j)=({x},{y})")

        plt.savefig("KitaevChain_spatial_corr_ED.png", dpi = 200, bbox_inches = "tight")
        plt.close()


