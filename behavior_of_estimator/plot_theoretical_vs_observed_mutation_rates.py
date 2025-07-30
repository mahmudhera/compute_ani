from matplotlib import pyplot as plt
import argparse 
import pandas as pd
import subprocess
import numpy as np

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


def expectation_p_hat_using_pmf(L, k, p):
    # invoke program ./nmut_pdf <L> <k> <p> <filename>
    # after program exits, the file will contain expectation of (L-Nmut)^(1/k)
    # we need to return 1 - (that expectation)/L^(1/k)
    
    intermediate_filename = f'expectation_values/for_{L}_{k}_{p}'
    with open(intermediate_filename, 'r') as f:
        expectation = float(f.read().strip())
    return 1 - expectation / (L**(1/k))
    


def parse_args():
    parser = argparse.ArgumentParser(description='Plot theoretical vs observed mutation rates.')
    # input file where estimated rates are written
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input file containing estimated mutation rates.')
    # L, k (integers)
    parser.add_argument('--L', type=int, required=True, help='Length of the DNA sequence.')
    parser.add_argument('--k', type=int, required=True, help='Length of the k-mer.')
    parser.add_argument('--o', type=str, required=True, help='Output filename (pdf extension).')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    estimated_rates_filename = args.input_file
    L, k = args.L, args.k
    
    estimated_rates = pd.read_csv(estimated_rates_filename)
    
    # filter for the given L and k
    filtered_rates = estimated_rates[(estimated_rates['L'] == L) & (estimated_rates['k'] == k)]
    
    true_p_to_estimated_average = {}
    true_p_to_estimated_std = {}
    true_p_list = filtered_rates['p'].unique()
    true_p_list.sort()
    for true_p in true_p_list:
        # get the average estimated rate for this true p
        avg_estimated_rate = filtered_rates[filtered_rates['p'] == true_p]['estimated_rate'].mean()
        std_estimated_rate = filtered_rates[filtered_rates['p'] == true_p]['estimated_rate'].std()
        true_p_to_estimated_average[true_p] = avg_estimated_rate
        true_p_to_estimated_std[true_p] = std_estimated_rate
        
    true_p_to_theoretical_expectation = {}
    for true_p in true_p_list:
        theoretical_expectation = expectation_p_hat(L, k, true_p)
        true_p_to_theoretical_expectation[true_p] = theoretical_expectation
        
        
    true_p_to_expectation_using_pmf = {}
    for true_p in true_p_list:
        expectation = expectation_p_hat_using_pmf(L, k, true_p)
        true_p_to_expectation_using_pmf[true_p] = expectation
        
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(true_p_list, list(true_p_to_theoretical_expectation.values()), marker='x', label='Theoretical Expectation of p_hat')
    plt.plot(true_p_list, list(true_p_to_expectation_using_pmf.values()), marker='s', label='Expectation using PMF')
    
    # Adding error bars for estimated rates 
    plt.errorbar(true_p_list, list(true_p_to_estimated_average.values()), 
                 yerr=list(true_p_to_estimated_std.values()), fmt='o', capsize=5, label='Estimated Rates with Std Dev')
    
    plt.xlabel('True Mutation Rate (p)')
    plt.ylabel('Mutation Rate')
    plt.title(f'Estimated vs Theoretical Mutation Rates for L={L}, k={k}')
    plt.legend()
    plt.grid()
    plt.savefig(args.o)
    