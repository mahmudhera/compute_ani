In this folder, I am investigating estimation of ANI between a pair of sequence files.

Comparisons between four index choices: (10K length string is tested, and 10K kmers are queried)

Index Build times (sec):
  BKTree: 0.13s
  HashNeighbor: 83.15s
  Trie: 0.08s
  FAISS-HNSW: 0.92s

✅ Accuracy Comparison:
Distance |   True |     BKTree | HashNeighb |       Trie | FAISS-HNSW
----------------------------------------------------------------------
       0 |   8118 |       8118 |       8118 |       8118 |       8118 | 
       1 |   1699 |       1699 |       1699 |       1699 |       1699 | 
       2 |    151 |        151 |        151 |        151 |        151 | 
       3 |     12 |         12 |          0 |         12 |         12 | 

⏱️ Query times (sec):
  BKTree: 122.63s
  HashNeighbor: 4.70s
  Trie: 91.38s
  FAISS-HNSW: 0.33s