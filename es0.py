import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

# Generate patterns
def generate_patterns(P, N, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.choice([-1, 1], size=(P, N))

# Compute overlaps m^{u,(n)} = (1/N) sum p^u_i * S_i
def compute_overlaps(S, patterns):
    return (patterns @ S) / S.size

# Update neuron state using cyclic Hopfield rule (from Ex. 0.2)
def update_state(S_prev, patterns, beta):
    P, N = patterns.shape
    overlaps = compute_overlaps(S_prev, patterns)
    input_sum = patterns[0] * overlaps[-1]
    for mu in range(1, P):
        input_sum += patterns[mu] * overlaps[mu - 1]
    return np.tanh(beta * input_sum)

# Exercise 0.4: Simulate and plot
def simulate_cyclic_hopfield(P=10, N=100, beta=4.0, nmax=20, seed=42):
    patterns = generate_patterns(P, N, seed=seed)
    S = patterns[0].copy()
    overlaps_over_time = np.zeros((nmax + 1, P))
    overlaps_over_time[0] = compute_overlaps(S, patterns)

    for n in range(1, nmax + 1):
        S = update_state(S, patterns, beta)
        overlaps_over_time[n] = compute_overlaps(S, patterns)
    
    plt.figure(figsize=(10, 5))
    for mu in range(P):
        plt.plot(overlaps_over_time[:, mu], label=f"$m^{{{mu+1}}}$")
    plt.xlabel("Iteration")
    plt.ylabel("Overlap")
    plt.title("Overlap Evolution Over Time")
    #plt.legend()
    plt.grid(True, axis='x')
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    simulate_cyclic_hopfield()