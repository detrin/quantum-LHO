# -*- coding: utf-8 -*-

import math
import sys

from math import factorial
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate

from tqdm import tqdm

# In case you want to plot without using X server.
# import matplotlib
# matplotlib.use('Agg') #changed the backend


args = list(map(lambda s: s.lower(), sys.argv))
coherent_arg = "--coherent" in args
lho_arg = "--lho" in args
show_arg = "--show" in args


def Norm(v):
    """Norm for eigenfunction of LHO."""
    return 1.0 / np.sqrt(np.sqrt(np.pi) * 2 ** v * factorial(v))


def make_Hr(n_max):
    """Return a list of np.poly1d objects representing Hermite polynomials."""
    if n_max == 0:
        return []
    # Define the Hermite polynomials up to order n_max by recursion:
    # H_[v] = 2qH_[v-1] - 2(v-1)H_[v-2]
    Hr = [None] * (n_max + 1)
    Hr[0] = np.poly1d([1.0, ])
    Hr[1] = np.poly1d([2.0, 0.0])
    for v in range(2, n_max + 1):
        Hr[v] = Hr[1] * Hr[v - 1] - 2 * (v - 1) * Hr[v - 2]
    return Hr


def get_psi_part(v, q):
    """Return the harmonic oscillator wavefunction for level v on coordinate q."""
    Hr = make_Hr(v + 1)
    return Norm(v) * Hr[v](q) * np.exp(-q * q / 2.0)


def get_psi(q, C):
    """Return the wavefunction for level v on coordinate q."""
    Hr_l = make_Hr(len(C))
    amp = 0
    for n in range(len(C)):
        amp += C[n] * 1.0 / np.sqrt(np.sqrt(np.pi)
                                    * 2 ** n * factorial(n)) * Hr_l[n](q)
    amp *= np.exp(-q * q / 2.0)
    return amp


def store_eigenfunctions(q_lin, n_max):
    """Store coefficients of LHO eigenfunctions on given coordinates"""
    Hr_l = make_Hr(n_max)
    eig_fun = np.zeros((q_lin.shape[0], len(C)), dtype="float64")
    for q_i in range(q_lin.shape[0]):
        for n in range(n_max):
            q = q_lin[q_i]
            eig_fun[q_i, n] = (
                1.0 / np.sqrt(np.sqrt(np.pi) * 2 ** n *
                              factorial(n)) * Hr_l[n](q)
            )
    return eig_fun


def psi_from_stored(q_lin, q_i, C, eig_fun):
    """Return the wavefunction form stored coefficients and weights."""
    q = q_lin[q_i]
    amp = 0.0
    for n in range(len(C)):
        amp += C[n] * eig_fun[q_i, n]
    amp *= np.exp(-q * q / 2.0)
    return amp


def gaussian(x, mu, sig, A):
    return (
        A
        * 1.0
        / (np.sqrt(2.0 * np.pi) * sig)
        * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    )


def rk4(f, t0, t1, y0, n, args):
    """"Runge-Kutta method https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods"""
    vt = np.zeros((n + 1), dtype="float64")
    vy = np.zeros((n + 1, y0.shape[0]), dtype=y0.dtype)
    h = (t1 - t0) / float(n)
    vt[0] = t = t0
    vy[0] = y = y0
    for i in range(1, n + 1):
        k1 = h * f(t, y, *args)
        k2 = h * f(t + 0.5 * h, y + 0.5 * k1, *args)
        k3 = h * f(t + 0.5 * h, y + 0.5 * k2, *args)
        k4 = h * f(t + h, y + k3, *args)
        vt[i] = t = t0 + i * h
        vy[i] = y = y + (k1 + k2 + k2 + k3 + k3 + k4) / 6
    return vt, vy


def dynamics(t, psi, H):
    """We evolve system with this system of differential equations in given basis."""
    return -1j * np.dot(H, psi)


# Building hamiltonian of Linear Harmonic Oscillator
# See https://en.wikipedia.org/wiki/Quantum_harmonic_oscillator

N = 42
annihilation_op = np.zeros((N, N), dtype="float64")
creation_op = np.zeros((N, N), dtype="float64")
for i in range(N):
    for j in range(N):
        if i + 1 == j:
            annihilation_op[i, j] = (i + 1) ** 0.5
        if i == j + 1:
            creation_op[i, j] = (j + 1) ** 0.5

omega = 1

x_op = (1 / 2.0 / omega) ** 0.5 * (creation_op + annihilation_op)
p_op = 1j * (omega / 2.0) ** 0.5 * (creation_op - annihilation_op)

H_LHO = 0.5 * np.dot(p_op, p_op) + 0.5 * omega ** 2 * np.dot(x_op, x_op)

# Setting animation writer
Writer = animation.writers["ffmpeg"]
writer = Writer(fps=30, metadata=dict(artist="Me"), bitrate=1800)

################################################################################
# Coherent states

if coherent_arg:
    t_max, t_count = 20, 500
    dt = t_max / float(t_count)

    # Choose alpha
    alp_size, alp_phase = 2, 0
    alpha = np.absolute(alp_size) * np.exp(1j * alp_phase)

    # Obtaining initial coefficients for coherent state
    C = np.zeros((N), dtype="complex128")
    for n in range(C.shape[0]):
        C[n] = (
            np.exp(-np.absolute(alpha) ** 2 / 2)
            * np.power(alpha, n)
            / math.sqrt(factorial(n))
        )

    # Setting initial wavefunction
    psi_0 = np.zeros((N), dtype="complex128")
    for i in range(N):
        psi_0[i] = C[i]
    psi_t = np.zeros((t_count + 1, N), dtype="complex128")

    # Running dynamics
    t_lin, psi_t = rk4(dynamics, 0, t_max, psi_0, t_count, args=(H_LHO,))

    fig, axes = plt.subplots(nrows=1)

    x_lin = np.linspace(-10, 10, 500)
    y_lin = np.zeros((t_count, x_lin.shape[0]), dtype="float64")
    t_lin = np.linspace(0, t_max, t_count)
    axes.set_ylim(-1.1, 1.1)

    eig_fun = store_eigenfunctions(x_lin, N)

    # This way the plotting won't be too slow
    def plot(ax, style):
        return ax.plot(x_lin, y_lin[0], style, animated=True)[0]

    def animate(t_i):
        lines[0].set_ydata(y_lin[t_i])
        return lines

    for t_i in tqdm(range(t_count)):
        for x_i in range(x_lin.shape[0]):
            x = x_lin[x_i]
            y_lin[t_i, x_i] = (
                np.absolute(
                    psi_from_stored(x_lin, x_i, np.real(psi_t[t_i]), eig_fun)
                    + 1j * psi_from_stored(x_lin, x_i,
                                           np.imag(psi_t[t_i]), eig_fun)
                )
                ** 2
            )

    lines = [plot(axes, "r-")]

    print("saving animation ... ")
    ani = animation.FuncAnimation(
        fig, animate, range(0, t_count), interval=30, blit=True
    )
    if show_arg:
        plt.show()
    else:
        ani.save("coherent_p.mp4", writer=writer)

    fig, axes = plt.subplots(nrows=1)

    x_lin = np.linspace(-10, 10, 500)
    y_lin_re = np.zeros((t_count, x_lin.shape[0]), dtype="float64")
    y_lin_im = np.zeros((t_count, x_lin.shape[0]), dtype="float64")
    t_lin = np.linspace(0, t_max, t_count)
    axes.set_ylim(-1.1, 1.1)

    def animate(t_i):
        lines[0].set_ydata(y_lin_re[t_i])
        lines[1].set_ydata(y_lin_im[t_i])
        return lines

    for t_i in tqdm(range(t_count)):
        for x_i in range(x_lin.shape[0]):
            x = x_lin[x_i]
            y_lin_re[t_i, x_i] = psi_from_stored(
                x_lin, x_i, np.real(psi_t[t_i]), eig_fun
            )
            y_lin_im[t_i, x_i] = psi_from_stored(
                x_lin, x_i, np.imag(psi_t[t_i]), eig_fun
            )

    lines = [plot(axes, "r-"), plot(axes, "b-")]

    print("saving animation ... ")
    ani = animation.FuncAnimation(
        fig, animate, range(0, t_count), interval=30, blit=True
    )
    if show_arg:
        plt.show()
    else:
        ani.save("coherent_re_im.mp4", writer=writer)

################################################################################

if lho_arg:
    t_max, t_count = 20, 500
    dt = t_max / float(t_count)

    # Obtaining Fourier coefficients of gaussian packet
    print("obtaining fourier coefficients ... ")
    Amp = 1
    psi_0 = np.zeros((N), dtype="complex128")
    def wave_gauss(q): return gaussian(q, -1, 1, Amp)
    intervals = np.linspace(-10, 10, 10)
    C = []
    for n in tqdm(range(N)):
        result = integrate.quad(
            lambda q: (get_psi_part(n, q) * (wave_gauss(q) - get_psi(q, C))),
            -1000,
            intervals[0],
        )
        c = result[0]
        for int_i in range(intervals.shape[0] - 1):
            result = integrate.quad(
                lambda q: (get_psi_part(n, q) *
                           (wave_gauss(q) - get_psi(q, C))),
                intervals[int_i],
                intervals[int_i + 1],
            )
            c += result[0]
        result = integrate.quad(
            lambda q: (get_psi_part(n, q) * (wave_gauss(q) - get_psi(q, C))),
            intervals[-1],
            1000,
        )
        c += result[0]
        C.append(c)
    C /= np.linalg.norm(C)
    # Setting initial wavefunction
    psi_0 = np.zeros((N), dtype="complex128")
    for i in range(N):
        psi_0[i] = C[i]
    psi_t = np.zeros((t_count + 1, N), dtype="complex128")
    t_lin, psi_t = rk4(dynamics, 0, t_max, psi_0, t_count, args=(H_LHO,))

    fig, axes = plt.subplots(nrows=1)

    # Preparing input for plotting
    x_lin = np.linspace(-10, 10, 500)
    y_lin = np.zeros((t_count, x_lin.shape[0]), dtype="float64")
    t_lin = np.linspace(0, t_max, t_count)
    axes.set_ylim(-1.1, 1.1)

    # Storing evaluated eigenfunctions
    eig_fun = store_eigenfunctions(x_lin, N)

    def plot(ax, style):
        return ax.plot(x_lin, y_lin[0], style, animated=True)[0]

    def animate(t_i):
        lines[0].set_ydata(y_lin[t_i])
        return lines

    # Evaluating points in graphs
    for t_i in tqdm(range(t_count)):
        for x_i in range(x_lin.shape[0]):
            x = x_lin[x_i]
            y_lin[t_i, x_i] = (
                np.absolute(
                    psi_from_stored(x_lin, x_i, np.real(psi_t[t_i]), eig_fun)
                    + 1j * psi_from_stored(x_lin, x_i,
                                           np.imag(psi_t[t_i]), eig_fun)
                )
                ** 2
            )

    lines = [plot(axes, "r-")]

    print("saving animation ... ")
    ani = animation.FuncAnimation(
        fig, animate, range(0, t_count), interval=30, blit=True
    )
    if show_arg:
        plt.show()
    else:
        ani.save("LHO_p.mp4", writer=writer)

    ################################################################################

    fig, axes = plt.subplots(nrows=1)

    x_lin = np.linspace(-10, 10, 500)
    y_lin_re = np.zeros((t_count, x_lin.shape[0]), dtype="float64")
    y_lin_im = np.zeros((t_count, x_lin.shape[0]), dtype="float64")
    t_lin = np.linspace(0, t_max, t_count)
    axes.set_ylim(-1.1, 1.1)

    def animate(t_i):
        lines[0].set_ydata(y_lin_re[t_i])
        lines[1].set_ydata(y_lin_im[t_i])
        return lines

    for t_i in tqdm(range(t_count)):
        for x_i in range(x_lin.shape[0]):
            x = x_lin[x_i]
            y_lin_re[t_i, x_i] = psi_from_stored(
                x_lin, x_i, np.real(psi_t[t_i]), eig_fun
            )
            y_lin_im[t_i, x_i] = psi_from_stored(
                x_lin, x_i, np.imag(psi_t[t_i]), eig_fun
            )

    lines = [plot(axes, "r-"), plot(axes, "b-")]

    print("saving animation ... ")
    ani = animation.FuncAnimation(
        fig, animate, range(0, t_count), interval=30, blit=True
    )
    if show_arg:
        plt.show()
    else:
        ani.save("LHO_re_im.mp4", writer=writer)
