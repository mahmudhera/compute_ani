from kmer_index import KmerIndex
import random

def create_random_string(length: int) -> str:
    """Generate a random DNA string of given length."""
    import random
    return ''.join(random.choice('ACGT') for _ in range(length))

def extract_kmers(sequence: str, k: int) -> list:
    """Extract all k-mers from a given sequence."""
    return [sequence[i:i + k] for i in range(len(sequence) - k + 1)]

if __name__ == "__main__":
    
    # create a random DNA sequence of length 100K
    random_sequence = create_random_string(100000)
    
    # extract k-mers of length 31 from the random sequence
    k = 31
    kmers = extract_kmers(random_sequence, k)
    
    # Initialize the KmerIndex with k and Flat index type
    index = KmerIndex(k, faiss_index_type="Flat")

    # Insert k-mers into the index
    for kmer in kmers:
        index.insert(kmer)
        
    # Query a random k-mer from the sequence
    random_kmer = random.choice(kmers)
    nearest_kmer, estimated_hd = index.query(random_kmer)
    print(f"Random k-mer: {random_kmer}")
    print(f"Nearest k-mer: {nearest_kmer}, Estimated Hamming Distance: {estimated_hd}")