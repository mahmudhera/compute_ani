In this folder, I am investigating estimation of ANI between a pair of sequence files.

Implemented a simple FAISS index in Python. Using this index, for 100K length genome, I observe the following behaviors when querying kmers from a mutated sequence (during the queries, I am recording number of kmers with 0 mutations):

- Flat Index: index building is quick. counts match exactly. time taken for querying is 311 seconds
- HNSW: index building is slow. There is 0.5% deviation. time taken for querying is 10.52 seconds

These numbers are much slower than just taking a set intersection! Therefore, we must come have good justification of doing all these extra work. Lets see.