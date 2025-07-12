from kmer_index import KmerIndex
import random
from tqdm import tqdm
import time

def create_random_string(length: int) -> str:
    """Generate a random DNA string of given length."""
    import random
    return ''.join(random.choice('ACGT') for _ in range(length))

def extract_kmers(sequence: str, k: int) -> list:
    """Extract all k-mers from a given sequence."""
    return [sequence[i:i + k] for i in range(len(sequence) - k + 1)]

# mutate a string for a fixed subst rate. record the mutation choices. return mutated string, and a list of integers, whose i-th index contain number of kmers with i mutations
def mutate_string(sequence: str, subst_rate: float, k) -> tuple:
    """Mutate a DNA sequence with a given substitution rate."""
    bases = 'ACGT'
    mutation_choices = []
    mutated_sequence = []

    for base in sequence:
        if random.random() < subst_rate:
            new_base = random.choice(bases.replace(base, ''))  # choose a different base
            mutated_sequence.append(new_base)
            mutation_choices.append(1)  # record a mutation
        else:
            mutated_sequence.append(base)
            mutation_choices.append(0)  # no mutation

    mutated_sequence = ''.join(mutated_sequence)
    
    num_kmers_with_mutations = [0 for _ in range(k + 1)]
    for i in range(len(mutated_sequence) - k + 1):
        num_mutations_this_kmer = sum(mutation_choices[i:i + k])
        num_kmers_with_mutations[num_mutations_this_kmer] += 1
        
    return mutated_sequence, num_kmers_with_mutations
    

if __name__ == "__main__":
    
    # create a random DNA sequence of length 100K
    random_sequence = create_random_string(100000)
    
    # mutate the sequence with a substitution rate of 0.01
    subst_rate = 0.01
    mutated_sequence, num_kmers_with_mutations = mutate_string(random_sequence, subst_rate, 31)
    print(f"Original sequence length: {len(random_sequence)}")
    print(f"Mutated sequence length: {len(mutated_sequence)}")
    print(f"Number of k-mers with 0 mutations: {num_kmers_with_mutations[0]}")
    print(f"Number of k-mers with 1 mutation: {num_kmers_with_mutations[1]}")
    print(f"Total kmers with mutations: {sum(num_kmers_with_mutations)}")
    
    
    # extract k-mers of length 31 from the random sequence
    k = 31
    kmers = extract_kmers(random_sequence, k)
    
    # Initialize the KmerIndex with k and Flat index type
    index = KmerIndex(k, faiss_index_type="HNSW32")

    # Insert k-mers into the index
    for kmer in kmers:
        index.insert(kmer)
        
    print(f"Inserted {index.num_kmers} k-mers into the index.")
    
    # query all kmers in the mutated sequence
    start_time = time.time()
    mutated_kmers = extract_kmers(mutated_sequence, k)
    num_kmers_0_mutations = 0
    for kmer in tqdm(mutated_kmers, desc="Querying mutated kmers"):
        _, estimated_hd = index.query(kmer)
        if estimated_hd == 0:
            num_kmers_0_mutations += 1
    end_time = time.time()
            
    print(f"Number of k-mers with 0 mutations in the mutated sequence (found using index): {num_kmers_0_mutations}")
    print(f"Time taken to query mutated k-mers: {end_time - start_time:.2f} seconds")
    
    kmer_set_orig_string = set(kmers)
    kmer_set_mutated_string = set(mutated_kmers)
    start_time = time.time()
    num_kmers_0_mutations_set = len(kmer_set_orig_string.intersection(kmer_set_mutated_string))
    end_time = time.time()
    print(f"Number of k-mers with 0 mutations in the mutated sequence (found using set intersection): {num_kmers_0_mutations_set}")
    print(f"Time taken to find intersection using sets: {end_time - start_time:.2f} seconds")
            