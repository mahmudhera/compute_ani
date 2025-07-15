from collections import defaultdict
import itertools
from tqdm import tqdm

class HashNeighborIndex:
    def __init__(self, k, d=1):
        self.k = k
        self.d = d
        self.table = defaultdict(list)

    def hamming(self, a, b):
        return sum(c1 != c2 for c1, c2 in zip(a, b))

    def _generate_neighbors(self, kmer):
        letters = 'ACGT'
        yield kmer
        for dist in range(1, self.d + 1):
            for positions in itertools.combinations(range(self.k), dist):
                for substitutions in itertools.product(letters, repeat=dist):
                    kmer_list = list(kmer)
                    for pos, sub in zip(positions, substitutions):
                        if kmer[pos] != sub:
                            kmer_list[pos] = sub
                    yield ''.join(kmer_list)

    def insert(self, kmer):
        for neighbor in self._generate_neighbors(kmer):
            self.table[neighbor].append(kmer)

    def query(self, kmer):
        best_dist = self.k + 1
        for candidate in self.table.get(kmer, []):
            dist = self.hamming(kmer, candidate)
            best_dist = min(best_dist, dist)
        return best_dist if best_dist <= self.d else -1  # -1 means not found within d

    def build(self, kmer_list):
        for kmer in tqdm(kmer_list):
            self.insert(kmer)
