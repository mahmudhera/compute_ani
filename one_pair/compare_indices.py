import random
import argparse
import time
from collections import Counter, defaultdict
from tqdm import tqdm
import numpy as np

from bk_tree_index import BKTree
from hash_neighbor_index import HashNeighborIndex
from faiss_index import FaissKmerIndex

from utils import random_dna, get_kmers, mutate, hamming

# Import the index classes
# Make sure the four index classes are available from separate files or copy-paste above

# --- Benchmark script ---

def main(args):
    random.seed(42)

    print(f"Generating random DNA sequence of length {args.length}...")
    original = random_dna(args.length)
    original_kmers = get_kmers(original, args.k)

    print(f"Building indexes with {len(original_kmers)} k-mers...")
    build_times = {}
    indexes = {}

    # BK-tree
    start = time.time()
    bk = BKTree()
    bk.build(original_kmers)
    build_times["BKTree"] = time.time() - start
    indexes["BKTree"] = bk
    print("BKTree built with {} nodes".format(len(bk.root.children)))

    # Hash+neighbor
    start = time.time()
    hn = HashNeighborIndex(k=args.k, d=2)
    hn.build(original_kmers)
    build_times["HashNeighbor"] = time.time() - start
    indexes["HashNeighbor"] = hn
    print("HashNeighbor built with {} entries".format(len(hn.table)))
    

    # Trie
    from trie_index import TrieIndex
    start = time.time()
    trie = TrieIndex()
    trie.build(original_kmers)
    build_times["Trie"] = time.time() - start
    indexes["Trie"] = trie
    print("Trie built with {} nodes".format(len(trie.root.children)))

    # FAISS
    start = time.time()
    faiss_index = FaissKmerIndex(args.k, use_hnsw=True)
    faiss_index.build(original_kmers)
    build_times["FAISS-HNSW"] = time.time() - start
    indexes["FAISS-HNSW"] = faiss_index
    print("FAISS-HNSW built with {} k-mers".format(len(faiss_index.kmer_to_id)))

    print("Build times (sec):")
    for name, t in build_times.items():
        print(f"  {name}: {t:.2f}s")

    print(f"\nMutating string with substitution rate {args.sub_rate}...")
    mutated = mutate(original, args.sub_rate)
    mutated_kmers = get_kmers(mutated, args.k)

    print("Calculating ground truth Hamming distances...")
    ground_truth_counts = Counter()
    gt_distances = []

    for orig_kmer, mut_kmer in zip(original_kmers, mutated_kmers):
        dist = hamming(orig_kmer, mut_kmer)
        gt_distances.append(dist)
        ground_truth_counts[dist] += 1

    print("\nQuerying all mutated k-mers against each index...")
    results = {}

    for name, index in indexes.items():
        print(f"Querying with {name}...")
        query_start = time.time()
        recovered_counts = Counter()

        for mut_kmer in tqdm(mutated_kmers):
            dist = index.query(mut_kmer)
            recovered_counts[dist] += 1

        query_time = time.time() - query_start
        results[name] = {
            "query_time": query_time,
            "counts": recovered_counts
        }

    print("\n Accuracy Comparison:")
    print(f"{'Distance':>8} | {'True':>6} | " + " | ".join([f"{name[:10]:>10}" for name in results]))
    print("-" * 70)

    max_d = max(ground_truth_counts.keys())
    for d in range(max_d + 1):
        line = f"{d:>8} | {ground_truth_counts[d]:>6} | "
        for name in results:
            line += f"{results[name]['counts'][d]:>10} | "
        print(line)

    print("\n Query times (sec):")
    for name, r in results.items():
        print(f"  {name}: {r['query_time']:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, required=True, help="Length of random DNA sequence")
    parser.add_argument("--k", type=int, required=True, help="k-mer size")
    parser.add_argument("--sub_rate", type=float, required=True, help="Substitution rate (0-1)")
    args = parser.parse_args()
    main(args)
