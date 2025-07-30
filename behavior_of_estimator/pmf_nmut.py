import numpy as np
from scipy.special import hyp2f1
import time
from tqdm import tqdm
import sys
from scipy.stats import norm


def r1_to_q(k,r1):
	#return 1-(1-r1)**k
	r1 = float(r1)
	q = 1-(1-r1)**k
	return float(q)

def exp_n_mutated(L,k,r1):
	q = r1_to_q(k,r1)
	return L*q

def var_n_mutated(L,k,r1,q=None):
	if (r1 == 0): return 0.0
	r1 = float(r1)
	if (q == None):
		q = r1_to_q(k,r1)
	varN = L*(1-q)*(q*(2*k+(2/r1)-1)-2*k) \
	     + k*(k-1)*(1-q)**2 \
	     + (2*(1-q)/(r1**2))*((1+(k-1)*(1-q))*r1-q)
	assert (varN>=0.0)
	return float(varN)

def exp_n_mutated_squared(L, k, p):
    return var_n_mutated(L, k, p) + exp_n_mutated(L, k, p) ** 2

def third_moment_nmut_exact(L,k,p):
    t1 = (L * (-2 + 3*L) * p**2 + 3 * (1 - p)**(2*k) * (2 + (-1 + k - L) * p * (2 + k * p - L * p)) - (1 - p)**k * (6 + p * (-6 + L * (-6 + p + 6 * L * p))))/(p**2)
    t2 = (-2 + 2 * k - L) * (-1 + 2 * k - L) * (2 * k - L) * (-1 + (1 - p)**k)**3
    t3 = (1/(p**3))*(-6 * (-1 + k)**2 * (k - L) * p**3 + 6 * (1 - p)**(3 * k) * (2 + (-2 + 2 * k - L) * p) + (1 - p)**(2 * k) * (-12 + 6 * (2 * k + L) * p + 6 * (4 * k**2 + 2 * (1 + L) - 3 * k * (2 + L)) * p**2 - (-1 + k) * k * (-2 + 4 * k - 3 * L) * p**3) + 6 * (-1 + k) * (1 - p)**k * p * (-2 + p * (2 - k + 2 * L + (k * (-2 + 3 * k - 3 * L) + L) * p)))
    t4 = 6 * (-1 + (1 - p)**k) * ((k + k**2 - 2 * k * L + (-1 + L) * L) * (-1 + 2 * (1 - p)**k) * hyp2f1(1, 2 + k - L, k - L, 1) + (k + k**2 - 2 * k * L + (-1 + L) * L) * (1 - p)**k * (-1 + p) * hyp2f1(1, 2 + k - L, k - L, 1 - p) - (-2 * k + 4 * k**2 + L - 4 * k * L + L**2) * ((-1 + 2 * (1 - p)**k) * hyp2f1(1, 1 + 2 * k - L, -1 + 2 * k - L, 1)- (-1 + p)**(2 * k) * hyp2f1(1, 1 + 2 * k - L, -1 + 2 * k - L, 1 - p)))
    return t1+t2+t3+t4


def perform_exhaustive_counting_modified(k, p):
    arr = np.zeros((2, k+1))
    arr[0, k] = (1-p)**k
    for i in range(1,k+1):
        arr[1, k-i] = p * (1-p)**(k-i)
    return arr


def get_nmut_pdf_modified(num_bases, k, p):
    # here, L means # of bases, length of the whole string
    arr = perform_exhaustive_counting_modified(k, p)
    curr = np.zeros((num_bases+1, k+1))
    next = np.zeros((num_bases+1, k+1))
    curr[0] = arr[0]
    curr[1] = arr[1]

    for l in tqdm(range(k+1, num_bases+1)):
        next.fill(0)
        row_sums = np.sum(curr, axis=1)
        for x in range(num_bases):
            if x > 0:
                total = row_sums[x - 1]
            else:
                total = 0
            next[x, 0] = total * p
            next[x, 1:k] = curr[x-1, 0:k-1] * (1 - p)
            next[x, k] = (curr[x, k-1] + curr[x, k]) * (1 - p)
        next, curr = curr, next
    return np.sum(curr, axis=1)


def get_nmut_pdf_using_normal_distribution(L, k, p):
    mu = exp_n_mutated(L, k, p)
    sigma = np.sqrt(var_n_mutated(L, k, p))
    pdf = np.zeros(L + 1)
    for i in range(L + 1):
        # pdf[i] equals (CMF of Normal at i+0.5) - (CMF of Normal at i-0.5)
        pdf[i] = norm.cdf(i + 0.5, loc=mu, scale=sigma) - norm.cdf(i - 0.5, loc=mu, scale=sigma)
        
    # normalize the pdf
    pdf /= np.sum(pdf)
    
    return pdf


if __name__ == '__main__':
    # take L, k, p as command line arguments
    if len(sys.argv) != 4:
        print("Usage: python pmf_nmut.py <L> <k> <p>")
        sys.exit(1)
        
    L = int(sys.argv[1])
    k = int(sys.argv[2])
    p = float(sys.argv[3])
    q = 1 - (1-p)**k
    
    # show the arguments
    print(f'L: {L}, k: {k}, p: {p}, q: {q}')

    
    print('Expectation from pdf:')
    start = time.time()
    pdf = get_nmut_pdf_modified(L+k-1, k, p)
    end = time.time()
    print("time needed: ", end-start)
    expectation = sum( [i*pdf[i] for i in range(L+1)] )
    print(expectation)

    print('--')

    print('2nd moment from formula:')
    print( exp_n_mutated_squared(L,k,p) )
    print('2nd moment from pdf:')
    expectation = sum( [i**2*pdf[i] for i in range(L+1)] )
    print(expectation)

    print('--')

    print('3rd moment from formula:')
    print( third_moment_nmut_exact(L,k,p) )
    print('3rd moment from pdf:')
    expectation = sum( [i**3*pdf[i] for i in range(L+1)] )
    print(expectation)
    
    # compute expectation of (L - Nmut)^(1/k)
    print('Expectation of (L - Nmut)^(1/k) from pdf:')
    expectation1 = sum( [(L - i)**(1/k) * pdf[i] for i in range(L+1)] )
    print (expectation1)
    
    print('------------------')
    print('Investigating approximation of PMF using Normal pdf:')
    pdf = get_nmut_pdf_using_normal_distribution(L, k, p)
    print('Expectation from normal pdf:')
    expectation = sum([i * pdf[i] for i in range(L + 1)])
    print(expectation)
    print('Expectation from formula:')
    print(exp_n_mutated(L, k, p))
    
    print('Variance from normal pdf:')
    variance = sum([i**2 * pdf[i] for i in range(L + 1)]) - expectation**2
    print(variance)
    print('Variance from formula:')
    print(var_n_mutated(L, k, p))
    
    # 3rd moment
    print('3rd moment from normal pdf:')
    third_moment = sum([i**3 * pdf[i] for i in range(L + 1)])
    print(third_moment)
    print('3rd moment from formula:')
    print(third_moment_nmut_exact(L, k, p))
    
    # expectation of (L - Nmut)^(1/k) from normal pdf
    print('Expectation of (L - Nmut)^(1/k) from normal pdf:')
    expectation1_normal = sum([(L - i)**(1/k) * pdf[i] for i in range(L + 1)])
    print (expectation1_normal)
    
    
    
    