import random
from Bio import SeqIO

def get_kmers_from_dna_string(dna_string, k):
    """Extract k-mers from a DNA string."""
    return [dna_string[i:i+k] for i in range(len(dna_string) - k + 1)]

def get_kmers_from_file(file_path, k):
    """Extract k-mers from a FASTA file."""
    kmers = []
    for record in SeqIO.parse(file_path, "fasta"):
        seq = str(record.seq)
        kmers.extend([seq[i:i+k] for i in range(len(seq) - k + 1)])
    return kmers

def random_dna(length):
    return ''.join(random.choices('ACGT', k=length))

def get_kmers(seq, k):
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]

def mutate(seq, rate):
    seq = list(seq)
    for i in range(len(seq)):
        if random.random() < rate:
            orig = seq[i]
            alt = random.choice([b for b in 'ACGT' if b != orig])
            seq[i] = alt
    return ''.join(seq)

def hamming(s1, s2):
    return sum(a != b for a, b in zip(s1, s2))