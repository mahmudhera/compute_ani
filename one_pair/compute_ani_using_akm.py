import argparse 
from utils import get_kmers_from_file
from faiss_index import FaissKmerIndex

def parse_args():
    # args: file1, file2, k
    parser = argparse.ArgumentParser(description="Compute ANI using approx kmer matches")
    parser.add_argument("file1", type=str, help="Path to first FASTA file")
    parser.add_argument("file2", type=str, help="Path to second FASTA file")
    parser.add_argument("--k", type=int, default=31, help="k-mer size (default: 31)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    kmers_file1 = get_kmers_from_file(args.file1, args.k)
    kmers_file2 = get_kmers_from_file(args.file2, args.k)
    
    # build faiss index using kmers_file1
    faiss_index = FaissKmerIndex(args.k, use_hnsw=True)
    for kmer in kmers_file1:
        faiss_index.insert(kmer)
        
    # query kmers_file2 against the index
    counts = [0 for _ in range(args.k + 1)]
    for kmer in kmers_file2:
        dist = faiss_index.query(kmer)
        counts[dist] += 1
        
    print(counts)
    
    
if __name__ == "__main__":
    main()