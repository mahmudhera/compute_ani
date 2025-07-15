class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.kmer = None

class TrieIndex:
    def __init__(self):
        self.root = TrieNode()
        self.k = None

    def insert(self, kmer):
        if self.k is None:
            self.k = len(kmer)
        node = self.root
        for c in kmer:
            node = node.children.setdefault(c, TrieNode())
        node.is_end = True
        node.kmer = kmer

    def query(self, kmer):
        best = [None, self.k + 1]

        def dfs(node, pos, mismatches):
            if mismatches >= best[1]:
                return
            if pos == self.k:
                if node.is_end:
                    best[0] = node.kmer
                    best[1] = mismatches
                return
            for c, child in node.children.items():
                dfs(child, pos + 1, mismatches + (c != kmer[pos]))

        dfs(self.root, 0, 0)
        return best[1]

    def build(self, kmer_list):
        for kmer in kmer_list:
            self.insert(kmer)
