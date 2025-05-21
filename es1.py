from es0 import *
import math

def check_correct_retrieval(nmax = 50, P = 50, N= 100, beta = 4.0, seed = 42):
    indices = []
    patterns = generate_patterns(P, N, seed=seed)
    S = patterns[0].copy()
    overlaps = np.zeros(P)
    overlaps = compute_overlaps(S, patterns)
    indices.append(np.argmax(overlaps))

    for n in range(1, nmax + 1):
        S = update_state(S, patterns, beta)
        overlaps = compute_overlaps(S, patterns)
        indices.append(np.argmax(overlaps))
        
        if (indices[-1] - indices[-2]) % P != 1:
            return 0 
        
    return 1 


def retrieval_probability(N, alpha_values, iter = 10):
    prob = np.zeros(len(alpha_values))
    k = 0
    for alpha in alpha_values:
        P = int(alpha*N)
        nmax = 2*P
        for ii in range(iter):
            if check_correct_retrieval(nmax, P, N, seed = None):
                prob[k] += 1
        prob[k] = prob[k]/iter
        k += 1
    
    return prob


# BONUS : Network that stores two cycles simultaneously
def update_state_multiple(S_prev, W, beta = 4.0):
    input_sum = W @ S_prev
    return np.tanh(beta * input_sum)
    
    
def simulate_multiple_cyclic_hopfield(patterns, initial_pattern, P=10, N=100, beta=4.0, nmax=20, seed=42):
    S = np.zeros((nmax, N))
    S[0,:] = initial_pattern.copy()
    overlaps_over_time = np.zeros((nmax, P))
    overlaps_over_time[0,:] = compute_overlaps(S[0,:], patterns)

    for n in range(1, nmax):
        S[n,:] = np.tanh(beta*(patterns[0,:]*overlaps_over_time[n-1, math.floor(P/2)-1] + overlaps_over_time[n-1, 0:math.floor(P/2)-1] @ patterns[1:math.floor(P/2),:])+
                    beta*(patterns[math.floor(P/2),:] * overlaps_over_time[n-1, P-1] + overlaps_over_time[n-1, math.floor(P/2):P-1] @ patterns[math.floor(P/2)+1:P,:]))
        overlaps_over_time[n,:] = compute_overlaps(S[n,:], patterns)
    
    plt.figure(figsize=(10, 5))
    for mu in range(P):
        plt.plot(overlaps_over_time[:, mu], label=f"$m^{{{mu+1}}}$")
    plt.xlabel("Iteration")
    plt.ylabel("Overlap")
    plt.grid(True, axis='x')
    plt.legend()
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.show()


def multiple_cycles(N, P):
    patterns = generate_patterns(P, N, seed=42)
    first_cycle = patterns[:P//2, :]
    second_cycle = patterns[P//2:, :]
    simulate_multiple_cyclic_hopfield(patterns, first_cycle[0], P, N)
    simulate_multiple_cyclic_hopfield(patterns, second_cycle[0], P, N)


if __name__ == "__main__":
    # 1.1 : Redo the simulation of 0.4 for P = 50 patterns
    simulate_cyclic_hopfield(P=50, N=100, beta=4.0, nmax=20, seed=42)
    # 1.3
    alpha_values = np.arange(0.05, 0.4 + 0.05, 0.05)
    prob1 = retrieval_probability(N = 100, alpha_values = alpha_values, iter = 10)
    prob2 = retrieval_probability(N = 1000, alpha_values = alpha_values, iter = 10)
    
    plt.plot(alpha_values, prob1, label="N = 100")
    plt.plot(alpha_values, prob2, label="N = 1000")
    plt.xlabel("Network Load (alpha)")
    plt.ylabel("Retrieval Probability")
    plt.title("Retrieval Probability vs Network Load")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    multiple_cycles(N = 500, P = 10)



# QUESTION 1.1

# When we increase the number of stored patterns from 10 to 50 in the cyclic Hopfield model, we are forcing 
# the network to memorize far more patterns. 
# Each pattern contributes to the overall structure of the weight matrix and, as more patterns are added, 
# they begin to interfere with each other. This interference becomes overwhelming increasing P.


# QUESTION 1.2 : 

# The correct sequence is one where the network steps through patterns cyclically, 
# advancing exactly by +1 modulo P each time. This condition handles also the wrap-around.


# QUESTION 1.3 

# Roughly the cycle capacity (i.e. where the retrieval probability starts to drop) 
# is α = 0.2 for the first network and α = 0.25 for the second one.


# Question 1.4:

# From the plot obtained in 1.3 we can notice that the more neurons there are in the network, the 
# sharper the transition becomes between retrieval and non-retrieval. This is because as the number of 
# neurons increases, random fluctuations in the network dynamics become less significant, making pattern 
# retrieval more stable and precise. This reduces variability and causes the transition between successful 
# retrieval and failure to become sharper. In smaller networks, noise has a stronger effect, leading to 
# a more gradual and irregular transition, while in larger networks, the shift occurs more abruptly as the 
# capacity limit is reached.
