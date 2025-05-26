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

def simulate_continuous_hopfield(P=10, N=100, beta=4.0, tau=5.0, dt=0.5, T=100.0, tau_delay=0.5, seed=42):
    patterns = generate_patterns(P, N, seed=seed)

    steps = int(T / dt)
    delay_steps = int(tau_delay / dt)

    x = np.zeros((steps + 1, N))
    x[:delay_steps + 1] = patterns[0]

    overlaps = np.zeros((steps + 1, P))
    overlaps[0] = compute_overlaps(x[0], patterns)

    for k in range(1, steps + 1):
        x_delay = x[k - delay_steps] if k - delay_steps >= 0 else x[0]
        overlaps_delay = compute_overlaps(x_delay, patterns)
        input_sum = np.zeros(N)
        for mu in range(P):
            input_sum += patterns[mu] * overlaps_delay[mu - 1]
        dxdt = (-x[k - 1] + np.tanh(beta * input_sum)) / tau
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

    steps = int(T / dt)
    delay_steps = int(tau_delay / dt)

    x = np.zeros((steps + 1, N))
    x[:delay_steps + 1] = patterns[0]

    overlaps = np.zeros((steps + 1, P))
    overlaps[0] = compute_overlaps(x[0], patterns)

    for k in range(1, steps + 1):
        x_delay = x[k - delay_steps] if k - delay_steps >= 0 else x[0]
        overlaps_delay = compute_overlaps(x_delay, patterns)
        input_sum = np.zeros(N)
        for mu in range(P):
            input_sum += patterns[mu] * overlaps_delay[mu - 1]
        dxdt = (-x[k - 1] + np.tanh(beta * input_sum)) / tau
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

def run_capacity_test(N_values=(100, 1000), alpha_values=np.arange(0.05, 0.45, 0.05),
                                    beta=4.0, tau=5.0, dt=0.5, trials=10, T_factor=2,
                                    tau_delay_factor=2, threshold=0.7, seed=42):
    results = {}

    for N in N_values:
        retrieval_rates = []

        for alpha in alpha_values:
            P = int(alpha * N)
            success_count = 0

            for trial in range(trials):
                np.random.seed(seed + trial)
                patterns = generate_patterns(P, N)
                tau_delay = tau_delay_factor * tau
                T = T_factor * P * tau_delay
                steps = int(T / dt)
                delay_steps = int(tau_delay / dt)

                x = np.zeros((steps + 1, N))
                x[:delay_steps + 1] = patterns[0]

                m = np.zeros((steps + 1, P))
                m[0] = (1 / N) * (patterns @ x[0])

                correct = True
                last_peak = 0
                time_from_last_peak = 0

                for k in range(steps):
                    k_prev = max(0, k - delay_steps)
                    input_sum = (
                        beta * (patterns[0] * m[k_prev, P - 1] +
                                m[k_prev, 0:P - 1] @ patterns[1:P])
                    )
                    dxdt = (-x[k] + np.tanh(input_sum)) / tau
                    x[k + 1] = x[k] + dt * dxdt
                    m[k + 1] = (1 / N) * (patterns @ x[k + 1])

                    new_peak = np.argmax(m[k + 1])
                    if np.max(m[k + 1]) < threshold:
                        time_from_last_peak += dt
                    else:
                        time_from_last_peak = 0

                    if new_peak == (last_peak + 1) % P:
                        last_peak = new_peak
                    elif new_peak != last_peak:
                        correct = False
                        break

                if correct:
                    success_count += 1

            retrieval_rates.append(success_count / trials)

        results[N] = retrieval_rates

    # Plot results
    plt.figure(figsize=(8, 5))
    for N in N_values:
        plt.plot(alpha_values, results[N], marker='o', label=f'N={N}')
    plt.xlabel("Load alfa = P/N")
    plt.ylabel("Retrieval Success Rate")
    plt.title("Cycle Retrieval Capacity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#BONUS
def simulate_mixture_initial_condition(P=10, N=100, beta=4.0, tau=5.0, dt=0.5, T=100.0,
                                       tau_delay_factor=2, M=2, seed=42):
    np.random.seed(seed)
    patterns = generate_patterns(P, N)

    tau_delay = tau_delay_factor * tau
    steps = int(T / dt)
    delay_steps = int(tau_delay / dt)

    # Initial condition: 1/2 * sum of first M patterns
    x0 = 0.5 * np.sum(patterns[:M], axis=0)

    x = np.zeros((steps + 1, N))
    x[:delay_steps + 1] = x0

    overlaps = np.zeros((steps + 1, P))
    overlaps[0] = compute_overlaps(x0, patterns)

    for k in range(1, steps + 1):
        x_delay = x[k - delay_steps] if k - delay_steps >= 0 else x0
        overlaps_delay = compute_overlaps(x_delay, patterns)

        input_sum = np.zeros(N)
        for mu in range(P):
            input_sum += patterns[mu] * overlaps_delay[mu - 1]  # cyclic index

        dxdt = (-x[k - 1] + np.tanh(beta * input_sum)) / tau
        x[k] = x[k - 1] + dt * dxdt
        overlaps[k] = compute_overlaps(x[k], patterns)

    t_vals = np.arange(steps + 1) * dt
    plt.figure(figsize=(10, 5))
    for mu in range(P):
        plt.plot(t_vals, overlaps[:, mu], label=f"$m^{{{mu+1}}}$")
    plt.xlabel("Time (ms)")
    plt.ylabel("Overlap")
    plt.title(f"Mixture Init (M={M} patterns) - Overlap Evolution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def simulate_three_independent_cycles(N=150, beta=4.0, tau=5.0, dt=0.5, T=100.0, tau_delay_factor=2, seed=42):
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
    x0 = np.sum([patterns[i] for i in idxs], axis=0) / 2

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
    plt.figure(figsize=(10, 5))
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
    simulate_continuous_hopfield()
    simulate_continuous_hopfield_with_delay()
    run_capacity_test()

    simulate_mixture_initial_condition(M=2)
    simulate_mixture_initial_condition(M=3)

    simulate_three_independent_cycles()