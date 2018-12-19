"""Summary
"""
from misc import diag_idcs
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mystic as mys
import numpy as np
import scipy.optimize as opt
import torch

use_cuda = torch.cuda.is_available()

class IonChain(object):

    """Class representing a trapped ion chain

    Attributes:
        b_x (2-D array of floats): Eigenvectors of the x normal modes (dims = nxn)
        b_z (2-D array of floats): Eigenvectors of the z normal modes (dims = nxn)
        beta (float): Ratio of trapping strengths
        charge (float): Charge of an ion
        delta_w_x (array of floats): Distance between x normal mode frequencies
            (len = n-1)
        delta_w_z (array of floats): Distance between z normal mode frequencies
            (len = n-1)
        equil_pos (2-D array of floats): Equilibrium position of ions (dims = 2xn)
        len_scale (float): A natural length scale for the ions
        lin (str): Direction in which ions are linear
        mass (float): Mass of an ion
        mean_delta_w_x (float): Mean distance between x normal mode frequencies
        mean_delta_w_z (float): Mean distance between z normal mode frequencies
        n (int): Number of ions
        omega (array of floats): Trapping strengths (len = 2)
        w_x (array of floats): Frequencies of the x normal modes (len = n)
        w_z (array of floats): Frequencies of the z normal modes (len = n)
    """

    def __init__(self, n, omega, **kwargs):
        """Initializes an instance of the IonChain class

        Args:
            n (int): Number of ions
            omega (array of floats): Trapping strengths (len = 2)

        Keyword Args:
            charge (float): Charge of an ion (default = 1.6021e-19)
            lin (str): Direction in which ions are linear (default = "z")
            mass (float): Mass of an ion (default = 2.8395e-25)
        """
        ic_params = {"charge": 1.6021e-19, "lin": "z", "mass": 2.8395e-25}
        ic_params.update(kwargs)

        self.__dict__.update(ic_params)
        self.n = n
        self.omega = np.array(omega, dtype=float)

        epsilon = 8.8541e-12
        pi = np.pi

        beta = self.omega[0] / self.omega[1]
        ls = (
            self.charge ** 2 / (2 * pi * epsilon * self.mass * self.omega[1] ** 2)
        ) ** (1 / 3)

        self.beta = beta
        self.len_scale = ls

        self.calc_equil_pos()
        self.calc_modes()
        pass

    def calc_equil_pos(self):
        """Calculated the equilibrium position of ions
        """
        beta = self.beta
        lin = self.lin
        ls = self.len_scale
        n = self.n

        def V(x, z):
            """Calculates the nondimensionalized potential of the system

            Args:
                x (array of floats): x coordinates of the ions (len = n)
                z (array of floats): z coordinates of the ions (len = n)

            Returns:
                float: Nondimensionalized potential of the system
            """
            V_coulomb = np.zeros(n)
            for i in range(n):
                xi = x[i]
                xj = x[i + 1 :]
                zi = z[i]
                zj = z[i + 1 :]
                V_coulomb[i] = np.sum(1 / np.sqrt((xi - xj) ** 2 + (zi - zj) ** 2))
            V_coulomb = np.sum(V_coulomb)

            V_harmonic = np.sum(beta ** 2 * x ** 2 + z ** 2)
            return V_coulomb + V_harmonic

        guess = np.linspace(-(n - 1) / 2, (n - 1) / 2, n)

        if lin == "z":
            x = np.zeros(n)
            potential = lambda z: V(x, z)
            z = opt.minimize(potential, guess).x
            z.sort()
            r = np.vstack((x, z))
        elif lin == "x":
            z = np.zeros(n)
            potential = lambda x: V(x, z)
            x = opt.minimize(potential, guess).x
            x.sort()
            r = np.vstack((x, z))
        else:
            potential = lambda r: V(r[:n], r[n : 2 * n])
            guess = np.append(guess, guess)
            r = opt.minimize(potential, guess).x.reshape((2, n))

        self.equil_pos = ls * r
        pass

    def calc_modes(self):
        """Calculates the normal modes of the system
        """
        beta = self.beta
        n = self.n
        omega = self.omega
        r = self.equil_pos / self.len_scale

        def dV2_dxidxj(x, z, i, j):
            """Calculates the i,j component of the x Hessian

            Args:
                x (array of floats): x coordinates of the ions (len = n)
                z (array of floats): z coordinates of the ions (len = n)
                i (int): First index of the x derivatives
                j (int): Second index of the x derivatives

            Returns:
                float: i,j component of the nondimensionalized x Hessian
            """
            if i == j:
                xi, zi = x[i], z[i]
                xj, zj = np.append(x[:i], x[i + 1 :]), np.append(z[:i], z[i + 1 :])
                return 2 * beta ** 2 + np.sum(
                    (2 * (xi - xj) ** 2 - (zi - zj) ** 2)
                    * ((xi - xj) ** 2 + (zi - zj) ** 2) ** (-5 / 2)
                )
            else:
                xi, zi = x[i], z[i]
                xj, zj = x[j], z[j]
                return ((zi - zj) ** 2 - 2 * (xi - xj) ** 2) * (
                    (xi - xj) ** 2 + (zi - zj) ** 2
                ) ** (-5 / 2)

        def dV2_dzidzj(x, z, i, j):
            """Calculates the i,j component of the z Hessian

            Args:
                x (array of floats): x coordinates of the ions (len = n)
                z (array of floats): z coordinates of the ions (len = n)
                i (int): First index of the z derivatives
                j (int): Second index of the z derivatives

            Returns:
                float: i,j component of the nondimensionalized z Hessian
            """
            if i == j:
                xi, zi = x[i], z[i]
                xj, zj = np.append(x[:i], x[i + 1 :]), np.append(z[:i], z[i + 1 :])
                return 2 + np.sum(
                    (2 * (zi - zj) ** 2 - (xi - xj) ** 2)
                    * ((xi - xj) ** 2 + (zi - zj) ** 2) ** (-5 / 2)
                )
            else:
                xi, zi = x[i], z[i]
                xj, zj = x[j], z[j]
                return ((xi - xj) ** 2 - 2 * (zi - zj) ** 2) * (
                    (xi - xj) ** 2 + (zi - zj) ** 2
                ) ** (-5 / 2)

        x, z = r
        rng = range(n)

        Hess_x = np.zeros((n, n))
        Hess_z = np.zeros((n, n))
        for i in rng:
            for j in rng:
                Hess_x[i, j] = dV2_dxidxj(x, z, i, j)
                Hess_z[i, j] = dV2_dzidzj(x, z, i, j)

        lambda_x, b_x = np.linalg.eig(Hess_x)
        lambda_x_order = lambda_x.argsort()
        lambda_x, b_x = lambda_x[lambda_x_order], b_x[:, lambda_x_order]
        w_x = np.sqrt(lambda_x * omega[1] ** 2 / 2)
        delta_w_x = w_x[1:] - w_x[:-1]
        mean_delta_w_x = np.mean(delta_w_x)

        self.w_x = w_x
        self.b_x = b_x
        self.delta_w_x = delta_w_x
        self.mean_delta_w_x = mean_delta_w_x

        lambda_z, b_z = np.linalg.eig(Hess_z)
        lambda_z_order = lambda_z.argsort()
        lambda_z, b_z = lambda_z[lambda_z_order], b_z[:, lambda_z_order]
        w_z = np.sqrt(lambda_z * omega[1] ** 2 / 2)
        delta_w_z = w_z[1:] - w_z[:-1]
        mean_delta_w_z = np.mean(delta_w_z)

        self.w_z = w_z
        self.b_z = b_z
        self.delta_w_z = delta_w_z
        self.mean_delta_w_z = mean_delta_w_z
        pass

    def plot_equil_pos(self, **kwargs):
        """Plots the equilibrium position of the ions

        Keyword Args:
            fig (matplotlib.figure.Figure): Figure for plot (default = plt.figure())
            idx (int): 3 digit integer representing position of the subplot
                (default = 111)

        Returns:
            matplotlib.axes._subplots.AxesSubplot: Axes of the plot
        """
        plot_params = {
            "fig": plt.figure() if "fig" not in kwargs.keys() else None,
            "idx": 111,
        }
        plot_params.update(kwargs)

        r = self.equil_pos / self.len_scale

        ax = plot_params["fig"].add_subplot(plot_params["idx"])
        ax.plot(r[1], r[0], "o", color="r")
        ax.set_xlabel(r"$\frac{z}{l}$")
        ax.set_ylabel(r"$\frac{x}{l}$")
        ax.set_xlim(1.1 * r.min(), 1.1 * r.max())
        ax.set_ylim(1.1 * r.min(), 1.1 * r.max())
        ax.set_aspect("equal")
        ax.grid(True, "major", ls="-")
        ax.minorticks_on()
        ax.grid(True, "minor", ls="--")
        return ax
        pass

    def plot_modes(self, **kwargs):
        """Plots the normal mode frequencies of the ions

        Keyword Args:
            fig (matplotlib.figure.Figure): Figure for plot (default = plt.figure())
            idx (int): 3 digit integer representing position of the subplot
                (default = 111)

        Returns:
            matplotlib.axes._subplots.AxesSubplot: Axes of the plot
        """
        plot_params = {
            "fig": plt.figure() if "fig" not in kwargs.keys() else None,
            "idx": 111,
        }
        plot_params.update(kwargs)

        rel_w_x = self.w_x / self.omega[1]
        rel_w_z = self.w_z / self.omega[1]

        ax = plot_params["fig"].add_subplot(plot_params["idx"])

        y = np.linspace(0, 1, 2)

        x = self.beta * np.ones(2)
        ax.plot(x, y, "-", color="r", label=r"$\omega_x$")

        x = np.ones(2)
        ax.plot(x, y, "-", color="b", label=r"$\omega_z$")

        x = np.hstack([w * np.ones(21) for w in rel_w_x])
        y = np.hstack([np.linspace(0, 1, 21) for w in rel_w_x])
        ax.plot(x, y, "x", color="r", label=r"$\omega^{(x)}_m$")

        x = np.hstack([w * np.ones(21) for w in rel_w_z])
        y = np.hstack([np.linspace(0, 1, 21) for w in rel_w_z])
        ax.plot(x, y, "o", color="b", label=r"$\omega^{(z)}_m$")

        ax.set_xlabel(r"frequency in units of $\omega_z$")
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.grid(True, "major", ls="-")
        ax.minorticks_on()
        ax.grid(True, "minor", ls="--")
        return ax

    pass


class SpinLattice(object):

    """Class representing a spin lattice

    Attributes:
        J (2-D array of floats): Interaction graph (dims = nxn)
        n (int): Number of spins
        v (array of floats): Vectorized interaction graph (len = n(n-1)/2)
    """

    def __init__(self, x):
        """Initializes an instance of the SpinLattice class

        Args:
            x (2-D or 1-D array of floats): Interaction graph or vectorized interaction
                graph (dims = nxn or len = n(n-1)/2)
        """
        x = np.array(x, dtype=float)

        if len(x.shape) == 2:
            self.J = x
            self.n = len(np.diag(x))
            idcs = np.triu_indices(self.n, 1)
            v = x[idcs]
            self.v = v
        else:
            self.v = x
            self.n = np.array(np.ceil(np.sqrt(len(x) * 2)), int)
            idcs = np.triu_indices(self.n, 1)
            J = np.zeros((self.n, self.n))
            J[idcs] = x
            self.J = J + np.transpose(J)
        pass

    def __add__(self, other):
        """Adds interaction graphs of 2 SpinLattices

        Args:
            other (eshutiqs.classes.SpinLattice): Some SpinLattice

        Returns:
            eshutiqs.classes.SpinLattice: SpinLattice with interaction graph equal to
                the sum between self's and other's interaction graphs
        """
        return SpinLattice(self.J + other.J)

    def __sub__(self, other):
        """Subtracts interaction graphs of 2 SpinLattices

        Args:
            other (eshutiqs.classes.SpinLattice): Some SpinLattice

        Returns:
            eshutiqs.classes.SpinLattice: SpinLattice with interaction graph equal to
                the difference between self's and other's interaction graphs
        """
        return SpinLattice(self.J - other.J)

    def __mul__(self, scalar):
        """Multiplies the interaction graph of a SpinLattice by some scalar

        Args:
            scalar (float): Some scalar

        Returns:
            eshutiqs.classes.SpinLattice: SpinLattice with interaction graph equal to
                self's interaction graph multiplied by the scalar
        """
        return SpinLattice(scalar * self.J)

    def __rmul__(self, scalar):
        """Multiplies the interaction graph of a SpinLattice by some scalar

        Args:
            scalar (float): Some scalar

        Returns:
            eshutiqs.classes.SpinLattice: SpinLattice with interaction graph equal to
                self's interaction graph multiplied by the scalar
        """
        return SpinLattice(self.J * scalar)

    def __truediv__(self, scalar):
        """Divides the interaction graph of a SpinLattice by some scalar

        Args:
            scalar (float): Some scalar

        Returns:
            eshutiqs.classes.SpinLattice: SpinLattice with interaction graph equal to
                self's interaction graph divided by the scalar
        """
        return SpinLattice(self.J / scalar)

    def rms(self):
        """Calculated the root-mean-square of the vectorized interaction graph

        Returns:
            float: Root-mean-square of vectorized interaction graph
        """
        return np.mean(np.sum(self.v ** 2))

    def norm(self, norm="L2"):
        """Calculates the norm of the vectorized interaction graph

        Args:
            norm (str or func): Type of norm or some norm function (default = "L2")

        Returns:
            float: norm of the vectorized interaction graph
        """
        norms = {
            "L1": lambda x: np.sum(np.abs(x)),
            "L2": lambda x: np.linalg.norm(x),
            "inf": lambda x: np.abs(x).max(),
        }
        if norm in norms.keys():
            norm = norms[norm]
        return norm(self.v)

    def normalize(self, norm="L2"):
        """Normalizes the SpinLattice

        Args:
            norm (str or func): Type of norm or some norm function (default = "L2")

        Returns:
            eshutiqs.classes.SpinLattice: SpinLattice with self's interaction graph
                after being normalized
        """
        return self / self.norm(norm)

    # def diag_rms(self):
    #     J = self.J
    #     idcs_dict = {i: np.array(diag_idcs(self.n, i)) for i in range(1, self.n)}
    #     diag_rms = np.zeros(self.n - 1)
    #     for i in range(1, self.n):
    #         diag_tensor = torch.zeros(J.shape, dtype=torch.float64, device=self.dev)
    #         diag_tensor[:, idcs_dict[i][0], idcs_dict[i][1]] = 1
    #         diag_rms[:, i - 1] = torch.einsum(
    #             "nij,nij->n", (torch.abs(J), diag_tensor)
    #         )
    #     self.diag_rms = diag_rms
    #     pass

    def plot_interaction_graph(self, **kwargs):
        """Plots the normal mode frequencies of the ions

        Keyword Args:
            fig (matplotlib.figure.Figure): Figure for plot (default = plt.figure())
            idx (int): 3 digit integer representing position of the subplot
                (default = 111)
            plot_type(str): Type of plot (default = "bar3d")

        Returns:
            matplotlib.axes._subplots.Axes3DSubplot or
            matplotlib.axes._subplots.AxesSubplot: Axes of the plot
        """
        plot3d_params = {
            "fig": plt.figure() if "fig" not in kwargs.keys() else None,
            "idx": 111,
            "plot_type": "bar3d",
        }
        plot3d_params.update(kwargs)

        n = self.n
        Z = self.J

        if plot3d_params["plot_type"] == "bar3d":
            ax = plot3d_params["fig"].add_subplot(plot3d_params["idx"], projection="3d")

            Z = np.transpose(Z)

            X, Y = np.meshgrid(np.linspace(0, n - 1, n), np.linspace(0, n - 1, n))

            X = X.flatten()
            Y = Y.flatten()
            Z = Z.flatten()

            W = Z - Z.min()
            frac = W / W.max()
            norm = colors.Normalize(frac.min(), frac.max())
            color = cm.gist_rainbow(norm(frac))

            ax.bar3d(X, Y, np.zeros(len(Z)), 1, 1, Z, color=color)
            ax.set_xlabel(r"$i$")
            ax.set_ylabel(r"$j$")
            ax.set_zlabel(r"$J$")
            ax.set_xticks(np.linspace(0.5, n - 0.5, n))
            ax.set_xticklabels(range(n))
            ax.set_yticks(np.linspace(0.5, n - 0.5, n))
            ax.set_yticklabels(range(n))
            ax.set_xlim(0, n)
            ax.set_ylim(0, n)
            ax.set_zlim(min(0, 1.1 * Z.min()), 1.1 * Z.max())
        elif plot3d_params["plot_type"] == "imshow":
            ax = plot3d_params["fig"].add_subplot(plot3d_params["idx"])
            ax.imshow(Z, cmap=cm.gist_rainbow)
            ax.set_xlabel(r"$j$")
            ax.set_ylabel(r"$i$")

        cax = plt.cm.ScalarMappable(cmap=cm.gist_rainbow)
        cax.set_array(Z)
        cbar = plot3d_params["fig"].colorbar(cax, ax=ax)
        cbar.set_label(r"$J$")
        return ax

    pass


class SimulatedSpinLattice(SpinLattice):

    """Class representing the spin lattice simulated by a trapped ion chain

    Attributes:
        dir (str): Direction of normal modes used
        ic (eshutiqs.classes.IonChain): IonChain used
        J (2-D array of floats): Interaction graph
        k (float): Wavenumber of laser used
        mu (array of floats): Detunings used
        n (int): Number of spins
        Omega (2-D array of floats): Rabi frequencies used
        v (array of floats): vectortized interaction graph
    """

    def __init__(self, ic, mu, Omega, **kwargs):
        """Initializes an instance of the SimulatedSpinLattice class

        Args:
            ic (eshutiqs.classes.IonChain): IonChain used
            mu (array of floats): Detunings used (len = m)
            Omega (2-D array of floats): Rabi frequencies used (dims = nxm)

        Keyword Args:
            dir (str): Direction of normal modes used (default = "x")
            k (float): Wavenumber of laser used (default = 1.7699e7)
        """
        ssl_params = {"dir": "x", "k": 1.7699e7}
        ssl_params.update(kwargs)

        hbar = 1.0546e-34

        self.__dict__.update(ssl_params)
        self.ic = ic
        self.mu = np.array(mu, dtype=float)
        self.Omega = np.array(Omega, dtype=float)

        if self.dir == "x":
            w = self.ic.w_x
            b = self.ic.b_x

        else:
            w = self.ic.w_z
            b = self.ic.b_z

        eta = np.einsum("in,n->in", b, 2 * self.k * np.sqrt(hbar / (2 * ic.mass * w)))
        zeta = np.einsum("im,in->imn", self.Omega, eta)
        cmpl_kron_delta = 1 - np.identity(ic.n)
        J = np.einsum(
            "ij,imn,jmn,n,mn->ij",
            cmpl_kron_delta,
            zeta,
            zeta,
            w,
            1 / np.subtract.outer(self.mu ** 2, w ** 2),
        )

        super().__init__(J)
        pass

    def plot_detunings(self, **kwargs):
        """Plots the detunings used in SimulatedSpinLattice

        Keyword Args:
            fig (matplotlib.figure.Figure): Figure for plot (default = plt.figure())
            idx (int): 3 digit integer representing position of the subplot
                (default = 111)

        Returns:
            matplotlib.axes._subplots.AxesSubplot: Axes of the plot
        """
        plot_params = {
            "fig": plt.figure() if "fig" not in kwargs.keys() else None,
            "idx": 111,
        }
        plot_params.update(kwargs)

        if self.dir == "x":
            w = self.ic.w_x
        elif self.dir == "z":
            w = self.ic.w_z

        rel_w = w / self.ic.omega[1]
        rel_mu = self.mu / self.ic.omega[1]

        ax = plot_params["fig"].add_subplot(plot_params["idx"])

        y = np.linspace(0, 1, 2)

        for i in range(len(rel_w)):
            x = rel_w[i] * np.ones(2)
            if i == 0:
                ax.plot(x, y, "-", color="b", label=r"$\omega_m$")
            else:
                ax.plot(x, y, "-", color="b")

        for freq in rel_mu:
            x = freq * np.ones(2)
            ax.plot(x, y, "--", color="r")

        ax.set_xlabel(r"frequency in units of $\omega_z$")
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.grid(True, "major", ls="-")
        ax.minorticks_on()
        ax.grid(True, "minor", ls="--")
        ax.legend()
        return ax

    def plot_Rabi_freqs(self, **kwargs):
        """Plots the Rabi frequencies used in SimulatedSpinLattice

        Keyword Args:
            fig (matplotlib.figure.Figure): Figure for plot (default = plt.figure())
            idx (int): 3 digit integer representing position of the subplot
                (default = 111)
            plot_type(str): Type of plot (default = "bar3d")

        Returns:
            matplotlib.axes._subplots.Axes3DSubplot or
            matplotlib.axes._subplots.AxesSubplot: Axes of the plot
        """
        plot3d_params = {
            "fig": plt.figure() if "fig" not in kwargs.keys() else None,
            "idx": 111,
            "plot_type": "bar3d",
        }
        plot3d_params.update(kwargs)

        i, n = self.Omega.shape
        Z = self.Omega

        if plot3d_params["plot_type"] == "bar3d":
            ax = plot3d_params["fig"].add_subplot(plot3d_params["idx"], projection="3d")

            Z = np.transpose(Z)

            X, Y = np.meshgrid(np.linspace(0, i - 1, i), np.linspace(0, n - 1, n))

            X = X.flatten()
            Y = Y.flatten()
            Z = Z.flatten()

            W = Z - Z.min()
            if (W == 0).all():
                frac = W
            else:
                frac = W / W.max()
            norm = colors.Normalize(frac.min(), frac.max())
            color = cm.gist_rainbow(norm(frac))

            ax.bar3d(X, Y, np.zeros(len(Z)), 1, 1, Z, color=color)
            ax.set_xlabel(r"$i$")
            ax.set_ylabel(r"$n$")
            ax.set_zlabel(r"$\Omega$")
            ax.set_xticks(np.linspace(0.5, i - 0.5, i))
            ax.set_xticklabels(range(i))
            ax.set_yticks(np.linspace(0.5, n - 0.5, n))
            ax.set_yticklabels(range(n))
            ax.set_xlim(0, i)
            ax.set_ylim(0, n)
            ax.set_zlim(min(0, 1.1 * Z.min()), 1.1 * Z.max())
        elif plot3d_params["plot_type"] == "imshow":
            ax = plot3d_params["fig"].add_subplot(plot3d_params["idx"])
            ax.imshow(Z, cmap=cm.gist_rainbow)
            ax.set_xlabel(r"$n$")
            ax.set_ylabel(r"$i$")

        cax = plt.cm.ScalarMappable(cmap=cm.gist_rainbow)
        cax.set_array(Z)
        cbar = plot3d_params["fig"].colorbar(cax, ax=ax)
        cbar.set_label(r"$\Omega$")
        return ax

    pass


class BatchSimulatedSpinLattice(object):

    """Class representing a batch of spin lattices simulated by a trapped ion chain

    Attributes:
        bssl_params (dict): Some parameters of the BatchSimulatedSpinLattice class
        dev (torch.device): Device to store and perform computations on
        ic (eshutiqs.classes.IonChain): IonChain used
        J (3-D array of floats): Interaction graphs (dims = Nxnxn)
        N (int): size of the BatchSimulatedSpinLattice
        mu (2-D array of floats): Detunings used (dims = Nxm)
        n (int): Number of ions/spins
        nJ (3-D array of floats): Normalized interaction graphs (dims = Nxnxn)
        norm (array of floats): Norms of vectorized interaction graph
        nv (2-D array of floats): Normalized vectorized interaction graphs
            (dims = Nxn(n-1)/2)
        Omega (3-D array of floats): Rabi frequencies used (dims = Nxnxm)
        v (2-D array of floats): Vectorized interaction graphs (dims = Nxn(n-1)/2)
    """

    def __init__(self, ic, mu, Omega, **kwargs):
        """Initializes an instance of the BatchSimulatedSpinLattice class

        Args:
            ic (eshutiqs.classes.IonChain): IonChain used
            mu (array of floats): Detunings used (dims = Nxm)
            Omega (2-D array of floats): Rabi frequencies used (dims = Nxnxm)

        Keyword Args:
            dev (torch.device): Device to store and perform computations on
            dir (str): Direction of normal modes used (default = "x")
            k (float): Wavenumber of laser used (default = 1.7699e7)

        """
        bssl_params = {"dev": torch.device("cuda" if use_cuda else "cpu"), "dir": "x", "k": 1.7699e7}
        bssl_params.update(kwargs)

        hbar = 1.0546e-34

        self.__dict__.update(bssl_params)
        self.bssl_params = bssl_params
        self.ic = ic
        self.N = len(mu)
        self.mu = torch.tensor(mu).to(dtype=torch.float64, device=self.dev)
        self.n = ic.n
        self.Omega = torch.tensor(Omega).to(dtype=torch.float64, device=self.dev)

        if self.dir == "x":
            w = torch.tensor(self.ic.w_x).to(dtype=torch.float64, device=self.dev)

            b = torch.tensor(self.ic.b_x).to(dtype=torch.float64, device=self.dev)

        else:
            w = torch.tensor(self.ic.w_z).to(dtype=torch.float64, device=self.dev)

            b = torch.tensor(self.ic.b_z).to(dtype=torch.float64, device=self.dev)

        eta = torch.einsum(
            "in,n->in", (b, 2 * self.k * torch.sqrt(hbar / (2 * ic.mass * w)))
        )
        zeta = torch.einsum("pim,in->pimn", (self.Omega, eta))
        chi = torch.tensor(1 - np.identity(ic.n)).to(
            dtype=torch.float64, device=self.dev
        )

        J = torch.einsum(
            "ij,pimn,pjmn,n,mn->pij",
            (
                chi,
                zeta,
                zeta,
                w,
                1
                / torch.tensor(np.subtract.outer(self.mu ** 2, w ** 2)).to(
                    dtype=torch.float64, device=self.dev
                ),
            ),
        )

        self.J = J
        idcs = np.triu_indices(self.n, 1)
        self.v = self.J[:, idcs[0], idcs[1]]
        pass

    def __len__(self):
        """Size of BatchSimulatedSpinLattice

        Returns:
            int: Size of BatchSimulatedSpinLattice
        """
        return self.N

    def norm(self, norm="L2"):
        """Calculates the norm of the vectorized interaction graphs

        Args:
            norm (str or func): Type of norm or some norm function (default = "L2")
        """
        norms = {
            "L1": lambda x: torch.abs(x).sum(1),
            "L2": lambda x: torch.sqrt((x ** 2).sum(1)),
            "inf": lambda x: torch.abs(x).max(1)[0],
        }
        if norm in norms.keys():
            norm = norms[norm]
        self.norm = norm(self.v)
        return self.norm

    def normalize(self, norm="L2"):
        """Normalizes all the interaction graphs in BatchSimulatedSpinLattice

        Args:
            norm (str or func): Type of norm or some norm function (default = "L2")
        """
        self.norm(norm)
        self.nJ = torch.einsum("nij,n->nij", (self.J, 1 / self.norm))
        self.nv = torch.einsum("ni,n->ni", (self.v, 1 / self.norm))
        return self.nv

    def rms(self):
        self.rms = torch.sqrt((self.v ** 2).mean(1))
        pass

    def diag_rms(self):
        J = self.J
        idcs_dict = {i: np.array(diag_idcs(self.n, i)) for i in range(1, self.n)}
        diag_rms = torch.zeros((self.N, self.n - 1))
        for i in range(1, self.n):
            diag_tensor = torch.zeros(J.shape, dtype=torch.float64, device=self.dev)
            diag_tensor[:, idcs_dict[i][0], idcs_dict[i][1]] = 1
            diag_rms[:, i - 1] = torch.sqrt(
                torch.einsum("nij,nij,nij->n", (J, J, diag_tensor)) / (self.n - i)
            )
        self.nd = torch.einsum(
            "ni,n->ni", (diag_rms, 1 / torch.sqrt((diag_rms ** 2).sum(1)))
        )
        pass

    def datafy(self, *args):
        """Converts BatchSimulatedSpinLattice into a dictionary of data

        Returns:
            dict: Dictionary containing data of BatchSimulatedSpinLattice
        """
        keys = ["J", "mu", "nd", "nJ", "norm", "nv", "Omega", "rms", "v"]
        keys.append(args)

        bssl_data_dict = {}
        for key in keys:
            if key in self.__dict__.keys():
                bssl_data_dict[key] = self.__dict__[key]
        return bssl_data_dict

    pass
