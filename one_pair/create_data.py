from utils import random_dna, mutate

def create_data(length, subst_rate, num_mutated_files, file_prefix):
    """
    Create a set of FASTA files with a random DNA sequence and its mutated versions.
    
    Args:
        length (int): Length of the original DNA sequence.
        subst_rate (float): Substitution rate for mutations.
        num_mutated_files (int): Number of mutated files to create.
        file_prefix (str): Prefix for the output files.
    """
    original = random_dna(length)
    
    # Write the original sequence to a FASTA file
    with open(f"{file_prefix}_original.fasta", "w") as f:
        f.write(f">original\n{original}\n")
    
    # Create mutated versions
    for i in range(num_mutated_files):
        mutated = mutate(original, subst_rate)
        with open(f"{file_prefix}_mutated_{i+1}.fasta", "w") as f:
            f.write(f">mutated_{i+1}\n{mutated}\n")
            
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create original and mutated DNA sequences in FASTA format.")
    parser.add_argument("--length", type=int, required=True, help="Length of the original DNA sequence")
    parser.add_argument("--subst_rate", type=float, required=True, help="Substitution rate for mutations (0-1)")
    parser.add_argument("--num_mutated_files", type=int, default=10, help="Number of mutated files to create")
    parser.add_argument("--file_prefix", type=str, default="dna_sequence", help="Prefix for output files")
    
    args = parser.parse_args()
    
    create_data(args.length, args.subst_rate, args.num_mutated_files, args.file_prefix)