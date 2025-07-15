import numpy as np
import faiss

class FaissKmerIndex:
    def __init__(self, k, use_hnsw=False):
        self.k = k
        self.kmer_to_id = {}
        self.id_to_kmer = {}
        self.next_id = 0
        self.dim = 2 * k  # 2 bits per base (A,C,G,T)
        self.index = faiss.IndexHNSWFlat(self.dim, 32) if use_hnsw else faiss.IndexFlatL2(self.dim)

    def _encode(self, kmer):
        mapping = {'A': [0, 0], 'C': [0, 1], 'G': [1, 0], 'T': [1, 1]}
        return np.array([bit for c in kmer for bit in mapping[c]], dtype='float32')

    def insert(self, kmer):
        vec = self._encode(kmer).reshape(1, -1)
        self.kmer_to_id[kmer] = self.next_id
        self.id_to_kmer[self.next_id] = kmer
        self.index.add(vec)
        self.next_id += 1

    def query(self, kmer):
        vec = self._encode(kmer).reshape(1, -1)
        D, I = self.index.search(vec, 1)
        match = self.id_to_kmer[int(I[0][0])]
        return sum(c1 != c2 for c1, c2 in zip(kmer, match))

    def build(self, kmer_list):
        for kmer in kmer_list:
            self.insert(kmer)
