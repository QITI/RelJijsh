# imports
import matplotlib.pyplot as plt
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
        if self.configuration == "z-linear": # assumes equilibrium
            # configuration is linear in z, i.e., the x coordinates of
            # the trapped ions are 0 when in equilibrium
            potential = lambda z: V(np.zeros(n), z)
            guess = np.linspace(-(n - 1) / 2, (n - 1) / 2, n)
            z = opt.minimize(potential, guess).x
            r = np.vstack((np.zeros(n), z))
        elif self.configuration == "x-linear": # assumes equilibrium
            # configuration is linear in x, i.e., the z coordinates of
            # the trapped ions are 0 when in equilibrium
            potential = lambda x: V(x, np.zeros(n))
            guess = np.linspace(-(n - 1) / 2, (n - 1) / 2, n)
            x = opt.minimize(potential, guess).x
            r = np.vstack((x, np.zeros(n)))
        else: # no assumptions about the equilibrium position
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
        ax.minorticks_on()
        ax.grid(True, "minor", ls="--")
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fancybox=True,
            shadow=True)
        return ax


if __name__ == "__main__":
    # IonChain example
    n = 5
    omega = [5, 1]
    ic = IonChain(n, omega)
    omega_m = ic.x_freqs  # normal mode frequencies
    b_ij = ic.x_eigvecs  # normal mode eigenvectors
    
    print ("Helloccc")
    print(ic.n)
    ic.plot_equilibrium_position()
    ic.plot_frequencies()

    # incorrect assumption about the equilibrium configuration can lead
    # to errors

    # example 1
    # n = 5
    # omega = [5,1]
    # ic = IonChain(n,omega,configuration="x-linear")

    # example 2
    # n = 5
    # omega = [1,5]
    # ic = IonChain(n,omega,configuration="z-linear")

    # example 3
    # n = 12
    # omega = [5,1]
    # ic = IonChain(n,omega,configuration="z-linear")
    # ic = IonChain(n,omega,configuration="x-linear")
