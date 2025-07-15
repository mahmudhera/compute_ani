class BKTreeNode:
    def __init__(self, kmer):
        self.kmer = kmer
        self.children = {}

class BKTree:
    def __init__(self):
        self.root = None

    @staticmethod
    def hamming(a, b):
        return sum(c1 != c2 for c1, c2 in zip(a, b))

    def insert(self, kmer):
        if self.root is None:
            self.root = BKTreeNode(kmer)
            return
        node = self.root
        while True:
            dist = self.hamming(kmer, node.kmer)
            if dist in node.children:
                node = node.children[dist]
            else:
                node.children[dist] = BKTreeNode(kmer)
                break

    def query(self, kmer):
        best = [None, float('inf')]

        def recurse(node):
            dist = self.hamming(kmer, node.kmer)
            if dist < best[1]:
                best[0], best[1] = node.kmer, dist
            for d in range(dist - best[1], dist + best[1] + 1):
                if d in node.children:
                    recurse(node.children[d])

        if self.root:
            recurse(self.root)
        return best[1]

    def build(self, kmer_list):
        for kmer in kmer_list:
            self.insert(kmer)
