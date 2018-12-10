# imports
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import mystic as mys
import numpy as np
import scipy.optimize as opt


class IonChain(object):
    """Class representing a trapped ion chain

    Attributes:
        configuration (Str or None): Guess regarding the equilibrium
            configuration of the trapped ions (default = "z-linear")
        beta (Float): Ratio of the trapping strength in the x direction
            against the trapping strength in the z direction
        charge (Float): Charge of a trapped ion (default = 1.6021e-19)
        equilibrium_position (2-D Array of Floats): Equilibrium position
            of the trapped ions
        length_scale (Float): A natural length scale for the system
        mass (Float): Mass of a trapped ion (defualt = 2.8395e-25)
        n (Int): Number of trapped ions
        omega (Array of Floats): Trapping strength in the x and z
            directions
        x_eigvecs (2-D Array of Floats): Eigenvectors of the normal
            modes in the x direction
        x_freqs (Array of Floats): Frequencies of the normal modes in
            the x direction
        z_eigvecs (2-D Array of Floats): Eigenvectors of the normal
            modes in the z direction
        z_freqs (Array of Floats): Frequencies of the normal modes in
            the z direction
    """

    def __init__(self, n, omega, **kwargs):
        """Initializes an instance of the IonChain class

        Args:
            n (Int): Number of trapped ions
            omega (Array of Floats): Trapping strength in the x and z
                directions
        Keyword Args:
            configuration (Str or None): Guess regarding the equilibrium
                configuration of the trapped ions (defualt = "z-linear")
            charge (Float): Charge of a trapped ion (default =
                1.6021e-19)
            mass (Float): Mass of a trapped ion (default = 2.8395e-25)
        """
        assert len(omega) == 2

        # default keyword args
        ic_params = {
            "configuration": "z-linear",
            "charge": 1.6021e-19,
            "mass": 2.8395e-25
        }
        # update keyword args
        ic_params.update(kwargs)

        # assign attributes
        self.n = n
        self.omega = np.array(omega, dtype=float)
        self.__dict__.update(ic_params)

        # constants
        epsilon = 8.8541e-12  # permittivity
        pi = np.pi

        beta = omega[0] / omega[1]  # ratio of trappping strength

        # length scale is defined by the distance between the 2 ions in
        # a 2 trapped ion chain when in equilibrium assuming the system
        # is 1-D and the trapping strength is the trapping strength in
        # the z direction
        length_scale = (self.charge**2 / (
            2 * pi * epsilon * self.mass * self.omega[1]**2))**(1 / 3)

        # assign attributes
        self.beta = beta
        self.length_scale = length_scale

        # assign attributes using methods
        self.find_equilibrium_position()
        self.find_modes()
        pass

    def find_equilibrium_position(self):
        """Calculates the equilibrium position of the trapped ion chain
        """
        n = self.n

        def V(x, z):
            """Calculates the nondimensionalized potential of the system

            Args:
                x (Array of Floats): x coordinates of the trapped ions
                z (Array of Floats): z coordinates of the trapped ions

            Returns:
                Float: Nondimensionalized potential of the system
            """
            # calculate total nondimensionalized coulomb potential
            V_coulomb = np.zeros(n)
            for i in range(n):
                xi = x[i]
                xj = x[i + 1:]
                zi = z[i]
                zj = z[i + 1:]
                V_coulomb[i] = np.sum(1 / np.sqrt((xi - xj)**2 + (zi - zj)**2))
            V_coulomb = np.sum(V_coulomb)

            # calculate total nondimensionalized harmonic potential
            V_harmonic = np.sum(self.beta**2 * x**2 + z**2)
            return V_harmonic + V_coulomb

        # find nondimensionalized equilibrium position by minimizing
        # the nondimensionalized potential
        if self.configuration == "z-linear":  # assumes equilibrium
            # configuration is linear in z, i.e., the x coordinates of
            # the trapped ions are 0 when in equilibrium
            potential = lambda z: V(np.zeros(n), z)
            guess = np.linspace(-(n - 1) / 2, (n - 1) / 2, n)
            z = opt.minimize(potential, guess).x
            r = np.vstack((np.zeros(n), z))
        if self.configuration == "x-linear":  # assumes equilibrium
            # configuration is linear in x, i.e., the z coordinates of
            # the trapped ions are 0 when in equilibrium
            potential = lambda x: V(x, np.zeros(n))
            guess = np.linspace(-(n - 1) / 2, (n - 1) / 2, n)
            x = opt.minimize(potential, guess).x
            r = np.vstack((x, np.zeros(n)))
        else:  # no assumptions about the equilibrium position
            potential = lambda r: V(r[0:n], r[n:2 * n])
            guess = np.append(
                np.linspace(-(n - 1) / 2, (n - 1) / 2, n),
                np.linspace(-(n - 1) / 2, (n - 1) / 2, n),
            )
            r = opt.minimize(potential, guess).x
            r = r.reshape((2, n))

        # redimensionalize and assign attribute
        self.equilibrium_position = r * self.length_scale
        pass

    def find_modes(self):
        """Calculates the normal modes of the trapped ion chain
        """

        def dV2_dxidxj(x, z, i, j):
            """Calculates the value of the ith, jth element of the
            Hessian of the nondimensionalized potential with respect to
            the nondimensionalized x coordinates of the trapped ions

            Args:
                x (Array of Floats): x coordinates of the trapped ions
                z (Array of Floats): z coordinates of the trapped ions
                i (Int): First index representing the ion the x
                    derivative of the potential is taken with respect to
                j (Int): Second index representing the ion the x
                    derivative of the potential is taken with respect to

            Returns:
                Float: ith, jth element of the Hessian of the
                    nondimensionalized potential with respect to
                    nondimensionalized x coordinates of the trapped ions
            """
            if i == j:  # diagonal elements
                xi = x[i]
                xj = np.append(x[0:i], x[i + 1:])
                zi = z[i]
                zj = np.append(z[0:i], z[i + 1:])
                return 2 * self.beta**2 + np.sum(
                    (2 * (xi - xj)**2 - (zi - zj)**2) * (
                        (xi - xj)**2 + (zi - zj)**2)**(-5 / 2))
            else:  # non-diagonal elements
                xi = x[i]
                xj = x[j]
                zi = z[i]
                zj = z[j]
                return ((zi - zj)**2 - 2 * (xi - xj)**2) * (
                    (xi - xj)**2 + (zi - zj)**2)**(-5 / 2)

        def dV2_dzidzj(x, z, i, j):
            """Calculates the value of the ith, jth element of the
            Hessian of the nondimensionalized potential with respect to
            the nondimensionalized z coordinates of the trapped ions

            Args:
                x (Array of Floats): x coordinates of the trapped ions
                z (Array of Floats): z coordinates of the trapped ions
                i (Int): First index representing the ion the z
                    derivative of the potential is taken with respect to
                j (Int): Second index representing the ion the z
                    derivative of the potential is taken with respect to

            Returns:
                Float: ith, jth element of the Hessian of the
                    nondimensionalized potential with respect to
                    nondimensionalized z coordinates of the trapped ions
            """
            if i == j:  # diagonal elements
                xi = x[i]
                xj = np.append(x[0:i], x[i + 1:])
                zi = z[i]
                zj = np.append(z[0:i], z[i + 1:])
                return 2 + np.sum((2 * (zi - zj)**2 - (xi - xj)**2) * (
                    (zi - zj)**2 + (xi - xj)**2)**(-5 / 2))
            else:  # non-diagonal elements
                xi = x[i]
                xj = x[j]
                zi = z[i]
                zj = z[j]
                return ((xi - xj)**2 - 2 * (zi - zj)**2) * (
                    (zi - zj)**2 + (xi - xj)**2)**(-5 / 2)

        # calculate Hessians of the nondimensionalized potential when
        # system in equilibrium
        n = self.n
        r = self.equilibrium_position / self.length_scale
        x = r[0]
        z = r[1]
        Hessian_X = np.zeros((n, n))
        Hessian_Z = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                Hessian_X[i, j] = dV2_dxidxj(x, z, i, j)
                Hessian_Z[i, j] = dV2_dzidzj(x, z, i, j)

        # find eigenvalues and eigenvectors of the Hessians
        x_eigvals, x_eigvecs = np.linalg.eig(Hessian_X)
        x_order = x_eigvals.argsort()
        x_eigvals, x_eigvecs = x_eigvals[x_order], x_eigvecs[:, x_order]
        z_eigvals, z_eigvecs = np.linalg.eig(Hessian_Z)
        z_order = z_eigvals.argsort()
        z_eigvals, z_eigvecs = z_eigvals[z_order], z_eigvecs[:, z_order]

        # redimensionalize and get frequency of normal modes
        x_freqs = np.sqrt(x_eigvals * self.omega[1]**2 / 2)
        z_freqs = np.sqrt(z_eigvals * self.omega[1]**2 / 2)

        # assign attributes
        self.x_freqs = x_freqs
        self.x_eigvecs = x_eigvecs
        self.z_freqs = z_freqs
        self.z_eigvecs = z_eigvecs
        pass

    def plot_equilibrium_position(self, **kwargs):
        """Plots the equilibrium position of the trapped ion chain

        Keyword Args:
            fig (Figure): Figure to make the plot on
            index (Int): 3 digit integer representing position of the
                subplot on the figure

        Returns:
            Axes: Axes the equilibrium position is plotted on
        """
        # default keyword args
        plot_params = {"fig": plt.figure(), "index": 111}
        # update keyword args
        plot_params.update(kwargs)

        # nondimensionalize
        r = self.equilibrium_position / self.length_scale

        ax = plot_params["fig"].add_subplot(plot_params["index"])
        ax.plot(r[1], r[0], "o", color='r')
        ax.set_xlabel(r"$\frac{z}{l}$")
        ax.set_ylabel(r"$\frac{x}{l}$")
        ax.set_xlim(1.1 * r.min(), 1.1 * r.max())
        ax.set_ylim(1.1 * r.min(), 1.1 * r.max())
        ax.set_aspect("equal")
        ax.grid(True, "major", ls="-")
        ax.minorticks_on()
        ax.grid(True, "minor", ls="--")
        return ax

    def plot_frequencies(self, **kwargs):
        """Plots the normal mode frequencies of the trapped ion chain

        Keyword Args:
            fig (Figure): Figure to make the plot on
            index (Int): 3 digit integer representing position of the
                subplot on the figure

        Returns:
            Axes: Axes the normal mode frequencies is plotted on
        """
        # default keyword args
        plot_params = {"fig": plt.figure(), "index": 111}
        # update keyword args
        plot_params.update(kwargs)

        # nondimensionalize
        rel_x_freqs = self.x_freqs / self.omega[1]
        rel_z_freqs = self.z_freqs / self.omega[1]

        ax = plot_params["fig"].add_subplot(plot_params["index"])

        y = np.linspace(0, 1, 2)

        # plot trapping strength in the x direction
        x = self.beta * np.ones(2)
        ax.plot(x, y, "-", color="r", label=r"$\omega_x$")

        # plot trapping strength in the z direction
        x = np.ones(2)
        ax.plot(x, y, "-", color="b", label=r"$\omega_z$")

        # plot normal modes in the x direction
        x = np.hstack([freq * np.ones(21) for freq in rel_x_freqs])
        y = np.hstack([np.linspace(0, 1, 21) for freq in rel_x_freqs])
        ax.plot(x, y, "x", color="r", label=r"$\omega^{(x)}_m$")

        # plot normal modes in the z direction
        x = np.hstack([freq * np.ones(21) for freq in rel_z_freqs])
        y = np.hstack([np.linspace(0, 1, 21) for freq in rel_z_freqs])
        ax.plot(x, y, "o", color="b", label=r"$\omega^{(z)}_m$")

        ax.set_xlabel(r"frequency in units of $\omega_z$")
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.grid(True, "major", ls="-")


class SpinLattice(object):
    def __init__(self, x, n=None):
        x = np.array(x, dtype=float)

        assert len(x.shape) in [1, 2]

        if len(x.shape) == 2:
            assert x.shape[0] == x.shape[1]

            self.J = x
            self.n = x.shape[0]
            self.v = np.hstack([x[i, i + 1:] for i in range(self.n)])
        elif len(x.shape) == 1:
            assert len(x) == n * (n - 1) / 2

            self.v = x
            self.n = n
            J = np.zeros((n, n))
            k = 0
            for i in range(n):
                for j in range(i + 1, n):
                    J[i, j] = x[k]
                    J[j, i] = J[i, j]
                    k += 1
            self.J = J
        pass

    def __add__(self, other):
        assert self.n == other.n
        return SpinLattice(self.J + other.J)

    def __sub__(self, other):
        assert self.n == other.n
        return SpinLattice(self.J - other.J)

    def __mul__(self, alpha):
        return SpinLattice(alpha * self.J)

    def __rmul__(self, alpha):
        return SpinLattice(alpha * self.J)

    def __truediv__(self, alpha):
        assert alpha != 0
        return SpinLattice(self.J / alpha)

    def error(self, other, norm="inf-norm"):
        norm_dict = {"2-norm": self.norm, "inf-norm": self.inf_norm}
        assert norm_dict[norm]() != 0
        return (self - other).rms() / norm_dict[norm]()

    def rms(self):
        return np.sqrt(np.mean(self.v**2))

    def L1_norm(self):
        return np.sum(np.abs(self.v))

    def L2_norm(self):
        return np.linalg.norm(self.v)

    def inf_norm(self):
        return np.abs(self.v).max()

    def normalize(self, norm="L2-norm"):
        norm_dict = {
            "L1-norm": self.L1_norm,
            "L2-norm": self.L2_norm,
            "inf-norm": self.inf_norm,
        }

        if norm_dict[norm]() == 0:
            return self
        else:
            return self / norm_dict[norm]()

    def configure_u(self, u=None):
        if u is None:
            u = np.ones(self.n)

        u = np.array(u, dtype=float)

        assert np.prod(u.shape) == self.n

        if u.shape != (self.n, 1):
            u = u.reshape((self.n, 1))

        if np.linalg.norm(u) == 0:
            return u
        else:
            return u / np.linalg.norm(u)

    def find_basis(self, ic, u=None, **kwargs):
        assert self.n == ic.n

        gsl_params = {"k": 1.7699e7, "direction": "x"}
        gsl_params.update(kwargs)

        assert gsl_params["direction"] in ["x", "z"]

        u = self.configure_u(u)

        if gsl_params["direction"] == "x":
            w = ic.x_freqs
        elif gsl_params["direction"] == "z":
            w = ic.z_freqs

        delta_w = np.array([w[i + 1] - w[i] for i in range(ic.n - 1)])
        min_delta_w = delta_w.min()
        gsls = [
            GeneratedSpinLattice(ic, [w[i] + min_delta_w * 1e-10], u,
                                 **gsl_params) for i in range(len(w) - 1)
        ]
        vs = [i.v for i in gsls]
        vs = np.transpose(vs)
        Q, R = np.linalg.qr(vs)
        nvs = np.transpose(Q)
        return nvs

    def remainder(self, ic, u=None, **kwargs):
        assert self.n == ic.n

        gsl_params = {"k": 1.7699e7, "direction": "x"}
        gsl_params.update(kwargs)

        assert gsl_params["direction"] in ["x", "z"]

        u = self.configure_u(u)
        nvs = self.find_basis(ic, u, **gsl_params)
        alphas = np.array([np.dot(v, i) for i in nvs])
        remainder = self.v - np.matmul(alphas, nvs)
        return SpinLattice(remainder, self.n)

    def projection(self, ic, u=None, **kwargs):
        assert self.n == ic.n

        gsl_params = {"k": 1.7699e7, "direction": "x"}
        gsl_params.update(kwargs)

        assert gsl_params["direction"] in ["x", "z"]
        return self - self.remainder(ic, u, **gsl_params)

    def find_minimum_error(self, ic, u=None, **kwargs):
        assert self.n == ic.n

        gsl_params = {"k": 1.7699e7, "direction": "x"}
        gsl_params.update(kwargs)

        assert gsl_params["direction"] in ["x", "z"]
        return self.error(self.projection(ic, u, **gsl_params))

    def plot_interaction_graph(self, **kwargs):
        plot3d_params = {
            "fig": plt.figure(),
            "index": 111,
            "plot_type": "bar3d"
        }
        plot3d_params.update(kwargs)

        n = self.n
        Z = self.J

        if plot3d_params["plot_type"] == "bar3d":
            ax = plot3d_params["fig"].add_subplot(
                plot3d_params["index"], projection="3d")

            Z = np.transpose(Z)

            X, Y = np.meshgrid(
                np.linspace(0, n - 1, n), np.linspace(0, n - 1, n))

            X = X.flatten()
            Y = Y.flatten()
            Z = Z.flatten()

            W = Z - Z.min()
            frac = W / W.max()
            norm = colors.Normalize(frac.min(), frac.max())
            color = cm.gist_rainbow(norm(frac))

            ax.bar3d(X, Y, np.zeros(len(Z)), 1, 1, Z, color=color)
            ax.set_xlabel("i")
            ax.set_ylabel("j")
            ax.set_zlabel("J")
            ax.set_xticks(np.linspace(0.5, n - 0.5, n))
            ax.set_xticklabels(range(n))
            ax.set_yticks(np.linspace(0.5, n - 0.5, n))
            ax.set_yticklabels(range(n))
            ax.set_xlim(0, n)
            ax.set_ylim(0, n)
            ax.set_zlim(min(0, 1.1 * Z.min()), 1.1 * Z.max())
        elif plot3d_params["plot_type"] == "image":
            ax = plot3d_params["fig"].add_subplot(plot3d_params["index"])
            ax.imshow(Z, cmap=cm.gist_rainbow)
            ax.set_xlabel("j")
            ax.set_ylabel("i")

        cax = plt.cm.ScalarMappable(cmap=cm.gist_rainbow)
        cax.set_array(Z)
        cbar = plot3d_params["fig"].colorbar(cax, ax=ax)
        cbar.set_label("J")
        return ax


class GeneratedSpinLattice(SpinLattice):
    """Class representing a virtual spin lattice generated by a trapped
    ion chain

    Attributes:
        direction (Str): Type of normal modes (default = "x")
        ionchain (IonChain): Ion chain
        k (Float): Wavenumber of laser (default = 1.7699e7)
        mu (Array of Floats): Detunings
        Omega (2-D Array of Floats): Rabi frequencies
    """

    def __init__(self, ic, mu, Omega, **kwargs):
        """Initializes an instance of the GeneratedSpinLattice class

        Args:
            ionchain (IonChain): Trapped ion chain
            mu (Array of Floats): Detunings
            Omega (2-D Array of Floats): Rabi frequencies

        Keyword Args:
            direction (Str): Type of normal modes (default = "x")
            k (Float): Wavenumber of laser (default = 1.7699e7)
        """
        # default keyword args
        gsl_params = {"k": 1.7699e7, "direction": "x"}
        # update keyword args
        gsl_params.update(kwargs)

        assert gsl_params["direction"] in ["x", "z"]

        # constants
        hbar = 1.0546e-34

        # number of ions in trapped ion chain
        n = ic.n

        # assign attributes
        self.ionchain = ic
        self.mu = np.array(mu, dtype=float)
        self.Omega = np.array(Omega, dtype=float)
        self.__dict__.update(gsl_params)

        assert self.Omega.shape == (n, len(self.mu))

        # select normal modes
        if self.direction == "x":
            b = ic.x_eigvecs
            w = ic.x_freqs
        elif self.direction == "z":
            b = ic.z_eigvecs
            w = ic.z_freqs

        # calculate Lamb-Dicke parameter, eta_i,l
        eta = np.zeros((n, n))
        for i in range(n):
            for l in range(n):
                eta[i, l] = b[i, l] * 2 * self.k * np.sqrt(
                    hbar / (2 * ic.mass * w[l]))

        # calculate F_i,j,m
        F = np.zeros((n, n, len(mu)))
        for i in range(n):
            for j in range(n):
                for m in range(len(mu)):
                    F[i, j, m] = np.sum(eta[i, :] * eta[j, :] * w[:] /
                                        (self.mu[m]**2 - w[:]**2))

        # calculate interaction graph, J_i,j
        J = np.zeros((n, n))
        for i in range(n): # diagonal elements
            J[i, i] = 0
        for i in range(n): # non-diagonal elements
            for j in range(i + 1, n):
                J[i, j] = np.sum(
                    self.Omega[i, :] * self.Omega[j, :] * F[i, j, :])
                J[j, i] = J[i, j]

        super().__init__(J)
        pass

    def plot_detunings(self, **kwargs):
        """Plots the detunings used to generate the virtual spin lattice

        Keyword Args:
            fig (Figure): Figure to make the plot on
            index (Int): 3 digit integer representing position of the
                subplot on the figure

        Returns:
            Axes: Axes the detunings are plotted on
        """
        plot_params = {"fig": plt.figure(), "index": 111}
        plot_params.update(kwargs)

        if self.direction == "x":
            w = self.ionchain.x_freqs
        elif self.direction == "z":
            w = self.ionchain.z_freqs

        rel_w = w / self.ionchain.omega[1]
        rel_mu = self.mu / self.ionchain.omega[1]

        ax = plot_params["fig"].add_subplot(plot_params["index"])

        y = np.linspace(0, 1, 2)

        for freq in rel_w:
            x = freq * np.ones(2)
            ax.plot(x, y, "-", color="b")

        for freq in rel_mu:
            x = freq * np.ones(2)
            ax.plot(x, y, "--", color="r")

        ax.set_xlabel("frequency")
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.grid(True, "major", ls="-")
        ax.minorticks_on()
        ax.grid(True, "minor", ls="--")
        return ax

    def plot_Rabi_frequencies(self, **kwargs):
        """Plots the Rabi frequencies used to generate the virtual spin
        lattice

        Keyword Args:
            fig (Figure): Figure to make the plot on
            index (Int): 3 digit integer representing position of the
                subplot on the figure

        Returns:
            Axes: Axes the Rabi frequencies are plotted on
        """
        plot3d_params = {
            "fig": plt.figure(),
            "index": 111,
            "plot_type": "bar3d"
        }
        plot3d_params.update(kwargs)

        i, n = self.Omega.shape
        Z = self.Omega

        if plot3d_params["plot_type"] == "bar3d":
            ax = plot3d_params["fig"].add_subplot(
                plot3d_params["index"], projection="3d")

            Z = np.transpose(Z)

            X, Y = np.meshgrid(
                np.linspace(0, i - 1, i), np.linspace(0, n - 1, n))

            X = X.flatten()
            Y = Y.flatten()
            Z = Z.flatten()

            ax = plot3d_params["fig"].add_subplot(
                plot3d_params["index"], projection="3d")

            W = Z - Z.min()
            frac = W / W.max()
            norm = colors.Normalize(frac.min(), frac.max())
            color = cm.gist_rainbow(norm(frac))

            ax.bar3d(X, Y, np.zeros(len(Z)), 1, 1, Z, color=color)
            ax.set_xlabel("i")
            ax.set_ylabel("n")
            ax.set_zlabel(r"$\Omega$")
            ax.set_xticks(np.linspace(0.5, i - 0.5, i))
            ax.set_xticklabels(range(i))
            ax.set_yticks(np.linspace(0.5, n - 0.5, n))
            ax.set_yticklabels(range(n))
            ax.set_xlim(0, i)
            ax.set_ylim(0, n)
            ax.set_zlim(min(0, 1.1 * Z.min()), 1.1 * Z.max())
        elif plot3d_params["plot_type"] == "image":
            ax = plot3d_params["fig"].add_subplot(plot3d_params["index"])
            ax.imshow(Z, cmap=cm.gist_rainbow)
            ax.set_xlabel("n")
            ax.set_ylabel("i")

        cax = plt.cm.ScalarMappable(cmap=cm.gist_rainbow)
        cax.set_array(Z)
        cbar = plot3d_params["fig"].colorbar(cax, ax=ax)
        cbar.set_label(r"$\Omega$")
        return ax
