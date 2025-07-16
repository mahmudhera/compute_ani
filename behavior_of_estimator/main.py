from matplotlib import pyplot as plt
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


def main():
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
                
    # for a fixed k and L, plot the expectation and error as a function of p
    k_fixed = 51
    L_fixed = 100000
    p_fixed = np.array(p_values)
    expectations = np.array([recorded_expectations[(L_fixed, k_fixed, p)]
                                for p in p_fixed])
    errors = np.array([recorded_errors[(L_fixed, k_fixed, p)]
                                for p in p_fixed])
    plt.figure(figsize=(12, 6))
    plt.plot(p_fixed, expectations, label='Expectation of p_hat', marker='o')
    plt.plot(p_fixed, errors, label='Error of p_hat', marker='x')
    plt.title(f'Expectation and Error of p_hat for L={L_fixed}, k={k_fixed}')
    plt.xlabel('p')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    plt.savefig('expectation_error_against_p.pdf')
    
    
if __name__ == "__main__":
    main()