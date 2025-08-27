import argparse 
from utils import get_kmers_from_file, get_kmers_from_dna_string
from faiss_index import FaissKmerIndex
import math

def parse_args():
    # args: file1, file2, k
    parser = argparse.ArgumentParser(description="Compute ANI using approx kmer matches")
    parser.add_argument("file1", type=str, help="Path to first FASTA file")
    parser.add_argument("file2", type=str, help="Path to second FASTA file")
    parser.add_argument("--k", type=int, default=31, help="k-mer size (default: 31)")
    
    return parser.parse_args()


def log_likelihood_function(p, k, kmer_count_vector):
    """
    Args:
        p: probability of nucleotide mutation (1 - ANI)
        k: k-mer size
        kmer_count_vector: vector of k-mer counts with k+1 entries, where
                kmer_count_vector[i] is the count of kmers with i mutations
    """
    if p < 0 or p > 1:
        raise ValueError("Probability p must be between 0 and 1")
    
    if len(kmer_count_vector) != k + 1:
        raise ValueError("kmer_count_vector must have k+1 entries")
    
    log_likelihood = 0.0
    for x in range(k + 1):
        # probability of a kmer having i mutations is k choose i * p^i * (1-p)^(k-i)
        p_x = math.comb(k, x) * (p ** x) * ((1 - p) ** (k - x))
        log_likelihood += kmer_count_vector[x] * math.log(p_x)
        
    return log_likelihood


def log_likelihood_function_merge_half_counts(p, k, kmer_count_vector):
    """
    Args:
        p: probability of nucleotide mutation (1 - ANI)
        k: k-mer size
        kmer_count_vector: vector of k-mer counts with k+1 entries, where
                kmer_count_vector[i] is the count of kmers with i mutations
    """
    if p < 0 or p > 1:
        raise ValueError("Probability p must be between 0 and 1")
    
    if len(kmer_count_vector) != k + 1:
        raise ValueError("kmer_count_vector must have k+1 entries")
    
    log_likelihood = 0.0
    probability_rest = 1.0
    for x in range(k + 1):
        # probability of a kmer having i mutations is k choose i * p^i * (1-p)^(k-i)
        p_x = math.comb(k, x) * (p ** x) * ((1 - p) ** (k - x))
        log_likelihood += kmer_count_vector[x] * math.log(p_x)
        probability_rest -= p_x
        
        if x == int(k/2):
            break
    
    if probability_rest > 0:
        log_likelihood += sum(kmer_count_vector[int(k/2)+1:]) * math.log(probability_rest)
        
    return log_likelihood


def compute_ani_from_kmer_counts_merge_half_counts(kmer_count_vector, k):
    """
    Compute ANI from k-mer counts using the log likelihood function.
    
    Args:
        kmer_count_vector: vector of k-mer counts with k+1 entries
        k: k-mer size
    Returns:
        ani: estimated Average Nucleotide Identity (ANI)
    """
    
    # Use a numerical optimization method to find the value of p that maximizes the log likelihood
    from scipy.optimize import minimize_scalar
    
    def objective_function(p):
        return -log_likelihood_function_merge_half_counts(p, k, kmer_count_vector)
    
    result = minimize_scalar(objective_function, bounds=(0, 1), method='bounded')
    
    if result.success:
        p_optimal = result.x
        ani = 1 - p_optimal
        return ani
    else:
        raise RuntimeError("Optimization failed to find a suitable p value")


def compute_ani_from_kmer_counts(kmer_count_vector, k):
    """
    Compute ANI from k-mer counts using the log likelihood function.
    
    Args:
        kmer_count_vector: vector of k-mer counts with k+1 entries
        k: k-mer size
    Returns:
        ani: estimated Average Nucleotide Identity (ANI)
    """
    
    # Use a numerical optimization method to find the value of p that maximizes the log likelihood
    from scipy.optimize import minimize_scalar
    
    def objective_function(p):
        return -log_likelihood_function(p, k, kmer_count_vector)
    
    result = minimize_scalar(objective_function, bounds=(0, 1), method='bounded')
    
    if result.success:
        p_optimal = result.x
        ani = 1 - p_optimal
        return ani
    else:
        raise RuntimeError("Optimization failed to find a suitable p value")
    
    
def compute_ani_from_dna_strings(dna_string1, dna_string2, k):
    """
    Computes the Average Nucleotide Identity (ANI) from two DNA strings using k-mers.
    
    Args:
        dna_string1: first DNA string
        dna_string2: second DNA string
        k: k-mer size
    Returns:
        ani: estimated Average Nucleotide Identity (ANI)
    """
    
    kmers1 = get_kmers_from_dna_string(dna_string1, k)
    kmers2 = get_kmers_from_dna_string(dna_string2, k)
    
    # build faiss index using kmers_file1
    faiss_index = FaissKmerIndex(k, use_hnsw=True)
    for kmer in kmers1:
        faiss_index.insert(kmer)
        
    # query kmers_file2 against the index
    counts = [0 for _ in range(k + 1)]
    for kmer in kmers2:
        dist = faiss_index.query(kmer)
        counts[dist] += 1
        
    # compute ANI from the kmer counts
    ani = compute_ani_from_kmer_counts_merge_half_counts(counts, k)
    
    return ani
        

def compute_ani(file1, file2, k):
    """
    Generates k-mers from two FASTA files and computes the Average Nucleotide Identity (ANI)
    using approximate k-mer matches.
    """
    kmers_file1 = get_kmers_from_file(file1, k)
    kmers_file2 = get_kmers_from_file(file2, k)
    
    # build faiss index using kmers_file1
    faiss_index = FaissKmerIndex(k, use_hnsw=True)
    for kmer in kmers_file1:
        faiss_index.insert(kmer)
        
    # query kmers_file2 against the index
    counts = [0 for _ in range(k + 1)]
    for kmer in kmers_file2:
        dist = faiss_index.query(kmer)
        counts[dist] += 1
        
    # compute ANI from the kmer counts
    ani = compute_ani_from_kmer_counts(counts, k)
    return ani



def compute_ani_from_kmer_counts_three_counts(num_kmers_0_mut, num_kmers_1_mut, num_kmers_2_or_more_mut, k):
    """
    Compute ANI from k-mer counts using the log likelihood function.
    
    Args:
        num_kmers_0_mut: count of kmers with 0 mutations
        num_kmers_1_mut: count of kmers with 1 mutation
        num_kmers_2_or_more_mut: count of kmers with 2 or more mutations
        k: k-mer size
    Returns:
        ani: estimated Average Nucleotide Identity (ANI)
    """
    
    # Use a numerical optimization method to find the value of p that maximizes the log likelihood
    from scipy.optimize import minimize_scalar
    
    def objective_function(p):
        p_0 = (1 - p) ** k
        p_1 = k * p * (1 - p) ** (k - 1)
        p_2_or_more = 1 - p_0 - p_1
        log_likelihood = (num_kmers_0_mut * math.log(p_0) +
                          num_kmers_1_mut * math.log(p_1) +
                          num_kmers_2_or_more_mut * math.log(p_2_or_more))
        return -log_likelihood
    
    result = minimize_scalar(objective_function, bounds=(0, 1), method='bounded')
    
    if result.success:
        p_optimal = result.x
        ani = 1 - p_optimal
        return ani
    else:
        raise RuntimeError("Optimization failed to find a suitable p value")
        


def main():
    args = parse_args()
    ani = compute_ani(args.file1, args.file2, args.k)
    
    print(f"Estimated ANI: {ani:.4f}")
    print(f"Estimated mutation rate: {1 - ani:.4f}")
    
    
if __name__ == "__main__":
    main()