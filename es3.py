import numpy as np
import matplotlib.pyplot as plt
import os
os.getcwd()
from es0 import *
from es2 import *

def compute_pca_components(x_time_series):
    x_centered = x_time_series - np.mean(x_time_series, axis=0)

    cov_matrix = np.cov(x_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    idx_sorted = np.argsort(eigvals)[::-1] #sort in decreasing order
    eigvals = eigvals[idx_sorted]
    eigvecs = eigvecs[:, idx_sorted]

    return eigvals, eigvecs

def simulate_for_pca(P=10, N=500, beta=4.0, tau=5.0, dt=0.5, T=500.0, tau_delay_factor=2, seed=42):

    np.random.seed(seed)
    patterns = generate_patterns(P, N)
    W = compute_weight_matrix(patterns)

    tau_delay = tau_delay_factor * tau
    steps = int(T / dt)
    delay_steps = int(tau_delay / dt)

    x = np.zeros((steps + 1, N))
    x[:delay_steps + 1] = patterns[0]  # Initialize at pattern 1

    for k in range(1, steps + 1):
        x_delay = x[k - delay_steps] if k - delay_steps >= 0 else x[0]
        dxdt = (-x[k - 1] + np.tanh(beta * (W @ x_delay))) / tau
        x[k] = x[k - 1] + dt * dxdt

    return x, patterns

def compute_loadings(x, eigvecs, K):
    return x @ eigvecs[:, :K]  # (T, N) × (N, K) = (T, K)

def reconstruct_state(x, eigvecs, K_values):
    reconstructions = {}
    for K in K_values:
        eigvecs_K = eigvecs[:, :K]         # (N, K)
        loadings_K = x @ eigvecs_K         # (T, K)
        x_hat = loadings_K @ eigvecs_K.T   # (T, N)
        reconstructions[K] = x_hat
    return reconstructions

#3.4
def plot_explained_variance(eigvals, max_components=20):
    total_var = np.sum(eigvals**2)
    pct_var = (eigvals[:max_components]**2) / total_var * 100  # percentage

    print("Explained variance by component:")
    for i, val in enumerate(pct_var, 1):
        print(f"Component {i}: {val:.2f}%")
    '''
    plt.figure(figsize=(8, 5))
    plt.bar(np.arange(1, max_components + 1), pct_var)
    plt.xlabel("Component Number")
    plt.ylabel("Explained Variance (%)")
    plt.title("Explained Variance per PCA Component")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()
    ''' 
    plt.figure(figsize=(8, 5))
    plt.bar(np.arange(1, max_components + 1), pct_var)

    # Increase font sizes
    plt.xlabel("Component Number", fontsize=15)
    plt.ylabel("Explained Variance (%)", fontsize=15)
    plt.title("Explained Variance per PCA Component", fontsize=16)

    # Increase tick label size
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # Run simulation and PCA for 3.2
    x, patterns = simulate_for_pca()
    eigvals, eigvecs = compute_pca_components(x)
    loadings = compute_loadings(x, eigvecs, K=10)

    # Define font sizes for reuse
    title_size = 16
    label_size = 14
    tick_size = 12
    legend_size = 12


    plt.figure(figsize=(10, 5))
    for k in range(loadings.shape[1]):
        plt.plot(loadings[:, k], label=f"$l_{{{k+1}}}(t)$")
    plt.xlabel("Time step", fontsize=label_size)
    plt.ylabel("Loading", fontsize=label_size)
    plt.title("PCA Loadings $l_k(t)$ for First 10 Components", fontsize=title_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.legend(fontsize=legend_size)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    P = patterns.shape[0]
    reconstructions = reconstruct_state(x, eigvecs, K_values=[P//2, P])
    neuron_idx = 0  # choose any neuron

    plt.figure(figsize=(10, 5))
    plt.plot(x[:, neuron_idx], label="Original", linewidth=2)
    for K, x_hat in reconstructions.items():
        plt.plot(x_hat[:, neuron_idx], label=f"Reconstructed (K={K})", linestyle='--')

    plt.xlabel("Time step", fontsize=label_size)
    plt.ylabel(f"Neuron {neuron_idx} activity", fontsize=label_size)
    plt.title(f"Reconstruction of Neuron {neuron_idx} with K = {P//2} and {P} components", fontsize=title_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)

    plt.legend(loc="upper right", fontsize=legend_size)
    plt.grid(True)
    plt.tight_layout()
    plt.show()




    '''
    # Plot 3.2
    plt.figure(figsize=(10, 5))
    for k in range(loadings.shape[1]):
        plt.plot(loadings[:, k], label=f"$l_{k+1}(t)$")
    plt.xlabel("Time step")
    plt.ylabel("Loading")
    plt.title("PCA Loadings $l_k(t)$ for First 10 Components")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 3.3 Reconstruct neuron activity for K = P/2 and K = P
    P = patterns.shape[0]
    print(P)
    reconstructions = reconstruct_state(x, eigvecs, K_values=[P//2, P])
    neuron_idx = 0  # choose any neuron

    plt.figure(figsize=(10, 5))
    plt.plot(x[:, neuron_idx], label="Original", linewidth=2)
    for K, x_hat in reconstructions.items():
        plt.plot(x_hat[:, neuron_idx], label=f"Reconstructed (K={K})", linestyle='--')
    plt.xlabel("Time step")
    plt.ylabel(f"Neuron {neuron_idx} activity")
    plt.title(f"Reconstruction of Neuron {neuron_idx} with K = {P//2} and {P} components")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    '''

    #3.4
    plot_explained_variance(eigvals, max_components=20)



# Theoretical comment 3.4:
# We observe that the first component explain around the 36% of the variance, the next few components
# still explain significant portion and after about the 6th component, the explained variance drops rapidly.
# The facts that only a few components capture most of rge variance indicates that the data lives in a lower
# dimensional space. We can conclude from the plot that the number of patterns P used in the network can be 
# roughly estimated from the cutoff point in the explained variance plot.

# Question 3.5:
# To estimate the dimensionality of recorded data using PCA, analyze the explained variance per component 
# and identify where the curve flattens out. The number of components before this flattening corresponds to the 
# intrinsic dimensionality (i.e., number of patterns or independent variables governing the dynamics).
