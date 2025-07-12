import numpy as np
import faiss


class KmerIndex:
    def __init__(self, k: int, faiss_index_type: str = "Flat"):
        self.k = k
        self.vector_dim = 4 * k
        self.faiss_index_type = faiss_index_type
        self.kmer_list = []

        # Build the FAISS index using the index factory
        self.index = faiss.index_factory(self.vector_dim, faiss_index_type)

        # Keep track of number of inserted vectors
        self.num_kmers = 0

    def _one_hot_encode(self, kmer: str) -> np.ndarray:
        """Encode k-mer to 4k-dimensional one-hot vector."""
        if len(kmer) != self.k:
            raise ValueError(f"K-mer must be of length {self.k}")

        base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        vec = np.zeros((self.vector_dim,), dtype=np.float32)

        for i, base in enumerate(kmer.upper()):
            if base not in base_map:
                raise ValueError(f"Invalid base '{base}' in k-mer")
            vec[4 * i + base_map[base]] = 1.0

        return vec.reshape(1, -1)

    def insert(self, kmer: str):
        """Insert k-mer into the FAISS index."""
        vec = self._one_hot_encode(kmer)
        self.index.add(vec)
        self.kmer_list.append(kmer)
        self.num_kmers += 1

    def query(self, kmer_query: str) -> str:
        """Query the nearest k-mer based on Hamming distance approximation."""
        if self.num_kmers == 0:
            raise ValueError("Index is empty. Insert kmers first.")

        query_vec = self._one_hot_encode(kmer_query)
        D, I = self.index.search(query_vec, 1)

        nearest_index = int(I[0][0])
        nearest_distance = D[0][0]

        # Optional: convert L2 distance to Hamming
        estimated_hd = int(round(nearest_distance / 2.0))

        return self.kmer_list[nearest_index], estimated_hd
