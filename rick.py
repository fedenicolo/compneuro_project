import numpy as np
import math

def generate_patterns(P: int, N: int):
    rng = np.random.default_rng()
    return rng.choice([-1, 1], size = (P,N))

def simulate_dynamics(P: int, N: int, n_max: int, beta: float, initial_pattern: int = 0, early_stopping: bool = False):
    S = np.zeros([n_max, N])
    m = np.zeros([n_max, P])
    correct = True

    patterns = generate_patterns(P, N)

    S[0,:] = patterns[initial_pattern,:]
    m[0,:] = 1/N * patterns @ S[0,:].T

    for n in range(1,n_max):
        S[n,:] = np.tanh(beta*(patterns[0,:]*m[n-1, P-1] + m[n-1, 0:P-1] @ patterns[1:P, :]))
        m[n,:] = 1/N * patterns @ S[n,:].T
        #Exe 1.2
        if np.argmax(m[n,:]) != (n + initial_pattern) % P:
            correct = False
            if early_stopping:
                break
    
    return m, S, correct

def check_instance(a: int, b: int, P: int):
    if a == P-1:
        return b == 0
    return b == a + 1


def check(m: np.ndarray, P: int):
    indeces = np.argmax(m, axis=1)
    for i in range(len(indeces)-1):
        if not check_instance(indeces[i], indeces[i+1], P):
            return False
    return True

def cycle_capacity(N: int, beta: float, alpha_vec: np.ndarray, num_simulations: int):
    correct_count = np.zeros(len(alpha_vec))
    for i, alpha in enumerate(alpha_vec):
        P = int(N*alpha)
        n_max = int(2*P)
        for r in range(num_simulations):
            _, _, correct = simulate_dynamics(P, N, n_max, beta, early_stopping=True)
            correct_count[i] += correct
    correct_count /= num_simulations
    return correct_count

def simulate_dynamics_2cycles(P: int, N: int, n_max: int, beta: float, initial_pattern: int = 0):
    S = np.zeros([n_max, N])
    m = np.zeros([n_max, P])
    correct = True

    patterns = generate_patterns(P, N)

    S[0,:] = patterns[initial_pattern,:]
    m[0,:] = 1/N * patterns @ S[0,:].T

    for n in range(1,n_max):
        S[n,:] = np.tanh(beta*(patterns[0,:]*m[n-1, math.floor(P/2)-1] + m[n-1, 0:math.floor(P/2)-1] @ patterns[1:math.floor(P/2), :]) + 
                         beta*(patterns[math.floor(P/2),:]*m[n-1, P-1] + m[n-1, math.floor(P/2):P-1] @ patterns[math.floor(P/2)+1:P, :]))
        m[n,:] = 1/N * patterns @ S[n,:].T
    return m, S

def simulate_continuous_dynamics(initial_pattern: int = 0, N: int = 100, P: int = 10, T: float = 100, dt: float = 0.5, tau: float = 5, tau_delay: float = 0.5, beta: float = 4, early_stopping: bool = False):
    correct = True
    time_from_last_peak = 0
    current_peak_duration = 0
    last_peak = initial_pattern
    patterns = generate_patterns(P, N)
    n_steps = int(np.floor(T/dt)) + 1
    t_vec = dt * np.arange(n_steps)
    x = np.zeros([n_steps, N])
    m = np.zeros([n_steps, P])
    x[0,:] = patterns[initial_pattern, :]
    m[0,:] = 1/N * patterns @ x[0,:].T 
    for k in range(0,n_steps-1):
        k_previous = int(max(0, k - np.ceil(tau_delay/dt)))
        x[k+1,:] = (1 - dt/tau) * x[k, :] + dt/tau * np.tanh(beta*(patterns[0,:]*m[k_previous, P-1] + m[k_previous, 0:P-1] @ patterns[1:P, :]))
        m[k+1,:] = 1/N * patterns @ x[k+1,:].T 
        new_peak = np.argmax(m[k+1,:])
        if max(m[k+1,:]) < 0.7:
            time_from_last_peak += dt
        else:
            time_from_last_peak = 0
        if new_peak == last_peak:
            current_peak_duration += dt
        elif new_peak == (last_peak + 1)%P:
            current_peak_duration = 0
        else:
            correct = False
            if early_stopping:
                break
        if time_from_last_peak > 3*tau_delay or current_peak_duration > 3*tau_delay:
            correct = False
            if early_stopping:
                break
        last_peak = new_peak
    return m, x, t_vec, correct

def continuous_cycle_capacity(N: int, beta: float, tau_delay: float, alpha_vec: np.ndarray, num_simulations: int):
    correct_count = np.zeros(len(alpha_vec))
    for i, alpha in enumerate(alpha_vec):
        P = int(N*alpha)
        T = int(2*P*tau_delay)
        for r in range(num_simulations):
            _, _, _, correct = simulate_continuous_dynamics(P = P, N = N, tau_delay = tau_delay, T = T, beta = beta, early_stopping=True)
            correct_count[i] += correct
    correct_count /= num_simulations
    return correct_count

def pca(X: np.ndarray):

    C = np.cov(X.T)
    eigenvalues, eigenvectors = np.linalg.eigh(C)

    indices = np.flip(np.argsort(np.abs(eigenvalues)))
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]


    return eigenvalues, eigenvectors

def explained_variance(eigenvalues: np.ndarray):

    return eigenvalues*2/sum(eigenvalues*2)

def simulate_continuous_dynamics_3_cycles(initial_patterns: np.ndarray, N: int = 100, P: int = 10, T: float = 100, dt: float = 0.5, tau: float = 5, tau_delay: float = 0.5, beta: float = 4):
    patterns = generate_patterns(P, N)
    n_steps = int(np.floor(T/dt)) + 1
    t_vec = dt * np.arange(n_steps)
    x = np.zeros([n_steps, N])
    m = np.zeros([n_steps, P])
    x[0,:] = 0.5*np.sum(patterns[initial_patterns, :], axis=0)
    m[0,:] = 1/N * patterns @ x[0,:].T 
    for k in range(0,n_steps-1):
        k_previous = int(max(0, k - np.ceil(tau_delay/dt)))
        x[k+1,:] = (1 - dt/tau) * x[k, :] + dt/tau * np.tanh(beta*(patterns[0,:]*m[k_previous, math.floor(P/3)-1] + m[k_previous, 0:math.floor(P/3)-1] @ patterns[1:math.floor(P/3), :]) + 
                         beta*(patterns[math.floor(P/3),:]*m[k_previous, 2*math.floor(P/3)-1] + m[k_previous, math.floor(P/3):2*math.floor(P/3)-1] @ patterns[math.floor(P/3)+1:2*math.floor(P/3), :]) +
                         beta*(patterns[2*math.floor(P/3),:]*m[k_previous, P-1] + m[k_previous, 2*math.floor(P/3):P-1] @ patterns[2*math.floor(P/3)+1:P, :]))
        m[k+1,:] = 1/N * patterns @ x[k+1,:].T 

    return m, x, t_vec