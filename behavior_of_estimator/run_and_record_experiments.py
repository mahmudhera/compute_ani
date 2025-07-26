from matplotlib import pyplot as plt
import numpy as np
import random
from tqdm import tqdm

from pmf_nmut import get_nmut_pdf_modified


def random_dna(length):
    return ''.join(random.choices('ACGT', k=length))

def get_kmers(seq, k):
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]

def mutate(seq, rate):
    seq = list(seq)
    for i in range(len(seq)):
        if random.random() < rate:
            orig = seq[i]
            alt = random.choice([b for b in 'ACGT' if b != orig])
            seq[i] = alt
    return ''.join(seq)


def variance_N_mut(L, k, p):
    if p == 0:
        return 0
    q = 1 - (1 - p)**k
    variance = L * (1-q) * ( q * ( 2*k+(2/p)-1 ) - 2*k) \
	            + k * (k-1) * (1-q)**2 \
	            + ( 2 * (1-q) / (p**2) ) * ( (1 + (k-1) * (1-q) )* p - q )       
    return variance



def expectation_p_hat (L, k, p):
    return min(p + variance_N_mut(L, k, p) * (1 - 1.0/k) * (1-p)**(1-2*k) / ( 2*k * L**2 ), 1.0)



def error_p_hat (L, k, p):
    return expectation_p_hat(L, k, p) - p


def compute_mut_rates(L, k, p, num_simulations):
    orig_str = random_dna(L)
    kmers_orig = get_kmers(orig_str, k)
    estimated_mutation_rates = []
    for _ in range(num_simulations):
        mutated_str = mutate(orig_str, p)
        kmers_mutated = get_kmers(mutated_str, k)
        num_intersection = len(set(kmers_orig) & set(kmers_mutated))
        p_hat = 1.0 - (num_intersection / len(kmers_orig))** (1.0 / k)
        estimated_mutation_rates.append(p_hat)
        
    return estimated_mutation_rates


def generate_theoretical_error():
    L_values = [10000, 100000, 1000000]
    k_values = [21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65]
    p_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4]
    
    recorded_errors = {}
    recorded_expectations = {}
    
    for L in L_values:
        for k in k_values:
            for p in p_values:
                exp_p_hat = expectation_p_hat(L, k, p)
                err_p_hat = error_p_hat(L, k, p)
                recorded_errors[(L, k, p)] = err_p_hat
                recorded_expectations[(L, k, p)] = exp_p_hat
                
    return recorded_errors, recorded_expectations


def plot_theoretical_error():
    recorded_errors, recorded_expectations = generate_theoretical_error()
                
    # for a fixed k and L, plot the expectation and error as a function of p
    k_fixed = 51
    L_fixed = 100000
    p_values = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4])
    
    expectations = np.array([recorded_expectations[(L_fixed, k_fixed, p)]
                                for p in p_values])
    errors = np.array([recorded_errors[(L_fixed, k_fixed, p)]
                                for p in p_values])
    estimated_mutation_rates = np.array([compute_mut_rate(L_fixed, k_fixed, p, 1000)[0]
                                        for p in tqdm(p_values)])
    
    plt.figure(figsize=(12, 6))
    plt.plot(p_values, expectations, label='Expectation of p_hat', marker='o')
    plt.plot(p_values, errors, label='Error of p_hat', marker='x')
    plt.plot(p_values, estimated_mutation_rates, label='Estimated Mutation Rate (uncorrected)', linestyle='--')
    plt.title(f'Expectation and Error of p_hat for L={L_fixed}, k={k_fixed}')
    plt.xlabel('p')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    plt.savefig('expectation_error_against_p.pdf')
    

def store_estimated_rates_in_file():
    L_list = [10000, 100000, 1000000]
    k_list = [21, 31, 41, 51, 61, 71]
    p_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
              0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2,
              0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29,
              0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37,
              0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5]
    num_simulations = 1000
    
    output_filename = 'mutation_rates.csv'
    f = open(output_filename, 'w')
    f.write('L,k,p,estimated_rate\n')
    
    for L in L_list:
        for k in k_list:
            for p in tqdm(p_list, desc=f'Processing L={L}, k={k}'):
                estimated_rates = compute_mut_rates(L, k, p, num_simulations)
                for rate in estimated_rates:
                    f.write(f"{L},{k},{p},{rate}\n")
                    
    f.close()
    
    
if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    store_estimated_rates_in_file()