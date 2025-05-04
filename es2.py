import numpy as np
import matplotlib.pyplot as plt

import os
os.getcwd()
from es0 import generate_patterns, compute_overlaps


def compute_weight_matrix(patterns):
    P, N = patterns.shape
    W = np.zeros((N, N))
    for mu in range(P):
        i_pattern = patterns[mu]
        j_pattern = patterns[mu - 1]  # previous pattern in the cycle (wraps around)
        W += np.outer(i_pattern, j_pattern)
    return W / N

# Exercise 2.1: Continuous dynamics simulation (no real delay yet)
def simulate_continuous_hopfield(P=10, N=100, beta=4.0, tau=5.0, dt=0.5, T=100.0, tau_delay=0.5, seed=42):
    patterns = generate_patterns(P, N, seed=seed)
    W = compute_weight_matrix(patterns)

    steps = int(T / dt)
    delay_steps = int(tau_delay / dt)

    # Initialize: x(t < 0) = p^1
    x = np.zeros((steps + 1, N))
    x[:delay_steps + 1] = patterns[0]

    overlaps = np.zeros((steps + 1, P))
    overlaps[0] = compute_overlaps(x[0], patterns)

    for k in range(1, steps + 1):
        x_delay = x[k - delay_steps] if k - delay_steps >= 0 else x[0]
        dxdt = (-x[k - 1] + np.tanh(beta * (W @ x_delay))) / tau
        x[k] = x[k - 1] + dt * dxdt
        overlaps[k] = compute_overlaps(x[k], patterns)


    t_vals = np.arange(steps + 1) * dt
    plt.figure(figsize=(10, 5))
    for mu in range(P):
        plt.plot(t_vals, overlaps[:, mu], label=f"$m^{{{mu+1}}}$")
    plt.xlabel("Time (ms)")
    plt.ylabel("Overlap")
    plt.title("Continuous Hopfield Overlap Evolution (No Delay)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def simulate_continuous_hopfield_with_delay(P=10, N=100, beta=4.0, tau=5.0, dt=0.5, T=100.0, seed=42):
    tau_delay = 2 * tau  
    patterns = generate_patterns(P, N, seed=seed)
    W = compute_weight_matrix(patterns)

    steps = int(T / dt)
    delay_steps = int(tau_delay / dt)

    x = np.zeros((steps + 1, N))
    x[:delay_steps + 1] = patterns[0]

    overlaps = np.zeros((steps + 1, P))
    overlaps[0] = compute_overlaps(x[0], patterns)

    for k in range(1, steps + 1):
        x_delay = x[k - delay_steps] if k - delay_steps >= 0 else x[0]
        dxdt = (-x[k - 1] + np.tanh(beta * (W @ x_delay))) / tau
        x[k] = x[k - 1] + dt * dxdt
        overlaps[k] = compute_overlaps(x[k], patterns)

    t_vals = np.arange(steps + 1) * dt
    plt.figure(figsize=(10, 5))
    for mu in range(P):
        plt.plot(t_vals, overlaps[:, mu], label=f"$m^{{{mu+1}}}$")
    plt.xlabel("Time (ms)")
    plt.ylabel("Overlap")
    plt.title(f"Continuous Hopfield with Delay tau_delay = {tau_delay} ms")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def run_capacity_test_continuous(N_values=(100, 1000), alpha_values=np.arange(0.05, 0.45, 0.05),
                                 beta=4.0, tau=5.0, dt=0.5, trials=10, T_factor=2,
                                 tau_delay_factor=2, seed=42):
    results = {}

    for N in N_values:
        retrieval_rates = []
        for alpha in alpha_values:
            P = int(alpha * N)
            success_count = 0
            for trial in range(trials):
                np.random.seed(seed + trial)
                patterns = generate_patterns(P, N)
                W = compute_weight_matrix(patterns)

                tau_delay = tau_delay_factor * tau
                T = T_factor * P * tau_delay
                steps = int(T / dt)
                delay_steps = int(tau_delay / dt)

                x = np.zeros((steps + 1, N))
                x[:delay_steps + 1] = patterns[0]
                overlaps = np.zeros((steps + 1, P))
                overlaps[0] = compute_overlaps(x[0], patterns)

                for k in range(1, steps + 1):
                    x_delay = x[k - delay_steps] if k - delay_steps >= 0 else x[0]
                    dxdt = (-x[k - 1] + np.tanh(beta * (W @ x_delay))) / tau
                    x[k] = x[k - 1] + dt * dxdt
                    overlaps[k] = compute_overlaps(x[k], patterns)

                # Process the max-overlap sequence
                max_indices = np.argmax(overlaps, axis=1)

                # De-duplicate consecutive values
                unique_seq = []
                for idx in max_indices:
                    if not unique_seq or idx != unique_seq[-1]:
                        unique_seq.append(idx)

                # Check if sequence [0, 1, ..., P-1] appears in order
                target = list(range(P))
                found = False
                for i in range(len(unique_seq) - P + 1):
                    if unique_seq[i:i+P] == target:
                        found = True
                        break

                if found:
                    success_count += 1

            retrieval_rates.append(success_count / trials)
        results[N] = retrieval_rates

    # Plot results
    plt.figure(figsize=(8, 5))
    for N in N_values:
        plt.plot(alpha_values, results[N], marker='o', label=f'N={N}')
    plt.xlabel("Load α = P/N")
    plt.ylabel("Retrieval Success Rate")
    plt.title("Cycle Retrieval Capacity (Continuous Model, Correct Detection)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def run_capacity_test_with_early_stopping(N_values=(100, 1000), alpha_values=np.arange(0.05, 0.45, 0.05),
                                beta=4.0, tau=5.0, dt=0.5, trials=10, T_factor=2,
                                tau_delay_factor=2, seed=42):
    results = {}
    
    for N in N_values:
        retrieval_rates = []
        for alpha in alpha_values:
            P = int(alpha * N)
            success_count = 0
            for trial in range(trials):
                np.random.seed(seed + trial)
                patterns = generate_patterns(P, N)
                W = compute_weight_matrix(patterns)

                tau_delay = tau_delay_factor * tau
                T = T_factor * P * tau_delay
                steps = int(T / dt)
                delay_steps = int(tau_delay / dt)

                x = np.zeros((steps + 1, N))
                x[:delay_steps + 1] = patterns[0]
                overlaps = np.zeros((steps + 1, P))
                overlaps[0] = compute_overlaps(x[0], patterns)

                max_indices = []

                for k in range(1, steps + 1):
                    x_delay = x[k - delay_steps] if k - delay_steps >= 0 else x[0]
                    dxdt = (-x[k - 1] + np.tanh(beta * (W @ x_delay))) / tau
                    x[k] = x[k - 1] + dt * dxdt
                    overlaps[k] = compute_overlaps(x[k], patterns)
                    max_indices.append(np.argmax(overlaps[k]))

                    # Optional: stop if simulation is very long and we have all patterns in order
                    observed_seq = []
                    for idx in max_indices:
                        if len(observed_seq) == 0 or idx != observed_seq[-1]:
                            observed_seq.append(idx)
                        if len(observed_seq) > P:
                            break

                    if observed_seq == list(range(P)):
                        success_count += 1
                        break

            retrieval_rates.append(success_count / trials)
        results[N] = retrieval_rates

    # Plot results
    plt.figure(figsize=(8, 5))
    for N in N_values:
        plt.plot(alpha_values, results[N], marker='o', label=f'N={N}')
    plt.xlabel("Load α = P/N")
    plt.ylabel("Retrieval Success Rate")
    plt.title("Cycle Retrieval Capacity (Corrected, Continuous Model)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


#BONUS
def simulate_mixture_initial_condition(P=10, N=100, beta=4.0, tau=5.0, dt=0.5, T=100.0,
                                       tau_delay_factor=2, M=2, seed=42):
    np.random.seed(seed)
    patterns = generate_patterns(P, N)
    W = compute_weight_matrix(patterns)
    
    tau_delay = tau_delay_factor * tau
    steps = int(T / dt)
    delay_steps = int(tau_delay / dt)

    # Initial condition: average of first M patterns
    x0 = np.mean(patterns[:M], axis=0)
    x0 = np.clip(x0, -1, 1)

    x = np.zeros((steps + 1, N))
    x[:delay_steps + 1] = x0

    overlaps = np.zeros((steps + 1, P))
    overlaps[0] = compute_overlaps(x0, patterns)

    for k in range(1, steps + 1):
        x_delay = x[k - delay_steps] if k - delay_steps >= 0 else x0
        dxdt = (-x[k - 1] + np.tanh(beta * (W @ x_delay))) / tau
        x[k] = x[k - 1] + dt * dxdt
        overlaps[k] = compute_overlaps(x[k], patterns)

    t_vals = np.arange(steps + 1) * dt
    plt.figure(figsize=(10, 5))
    for mu in range(P):
        plt.plot(t_vals, overlaps[:, mu], label=f"$m^{{{mu+1}}}$")
    plt.xlabel("Time (ms)")
    plt.ylabel("Overlap")
    plt.title(f"Mixture Init of M={M} Patterns — Overlap Evolution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def simulate_three_independent_cycles(N=150, beta=4.0, tau=5.0, dt=0.5, T=100.0, tau_delay_factor=2, seed=42):
    """
    Simulate a network with three independent cycles of different lengths.
    """
    np.random.seed(seed)
    
    # Define 3 cycles of different lengths
    cycle_lengths = [2, 3, 4]
    total_patterns = sum(cycle_lengths)
    P = total_patterns
    patterns = generate_patterns(P, N)

    # Build block-diagonal weight matrix for separate cycles
    W = np.zeros((N, N))
    segment_size = N // 3
    start_idx = 0

    for i, cl in enumerate(cycle_lengths):
        seg_start = i * segment_size
        seg_end = seg_start + segment_size
        for mu in range(cl):
            i_pattern = patterns[start_idx + mu]
            j_pattern = patterns[start_idx + (mu - 1) % cl]
            W[seg_start:seg_end, seg_start:seg_end] += np.outer(i_pattern[seg_start:seg_end], j_pattern[seg_start:seg_end])
        start_idx += cl

    W /= segment_size

    tau_delay = tau_delay_factor * tau
    steps = int(T / dt)
    delay_steps = int(tau_delay / dt)

    # Initial state: average of one pattern from each cycle
    idxs = [0, 2, 5]  # first pattern from each cycle
    x0 = np.mean([patterns[i] for i in idxs], axis=0)
    x0 = np.clip(x0, -1, 1)

    x = np.zeros((steps + 1, N))
    x[:delay_steps + 1] = x0
    overlaps = np.zeros((steps + 1, P))
    overlaps[0] = compute_overlaps(x0, patterns)

    for k in range(1, steps + 1):
        x_delay = x[k - delay_steps] if k - delay_steps >= 0 else x0
        dxdt = (-x[k - 1] + np.tanh(beta * (W @ x_delay))) / tau
        x[k] = x[k - 1] + dt * dxdt
        overlaps[k] = compute_overlaps(x[k], patterns)

    # Plot overlaps grouped by cycle
    t_vals = np.arange(steps + 1) * dt
    plt.figure(figsize=(12, 6))
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    start = 0
    for i, cl in enumerate(cycle_lengths):
        for mu in range(cl):
            idx = start + mu
            plt.plot(t_vals, overlaps[:, idx], label=f"Cycle {i+1} - $m^{{{idx+1}}}$", color=colors[i])
        start += cl
    plt.xlabel("Time (ms)")
    plt.ylabel("Overlap")
    plt.title("Three Independent Cycles — Overlap Evolution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    #simulate_continuous_hopfield()
    #simulate_continuous_hopfield_with_delay()
    #run_capacity_test_continuous()
    #run_capacity_test_with_early_stopping()

    #for the bonus
    # Run for M = 2 and M = 3
    #simulate_mixture_initial_condition(M=2)
    # simulate_mixture_initial_condition(M=3)

    simulate_three_independent_cycles()

#Es 2.1
#At t = 0, the overlap is close to 1, as expected.
#Over time, all overlap variables fluctuate weakly and irregularly.
#There is no clear cyclic pattern or consistent progression from one pattern to the next.
#The network seems to converge to a blended or ambiguous state, where all patterns are partially activated.

#Es 2.2
#The network now exhibits clear cyclic behavior: each overlap rises and falls in sequence.
#The cycle proceeds as: m1 -> m2 ... -> m10 -> m1 -> ...
#The cyclic pattern is stable and repeats cleanly, showing a limit cycle.

#Es 2.3
# from a point onwards, the success rate drops rapiddly to 0.