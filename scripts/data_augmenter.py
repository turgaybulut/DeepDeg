import argparse
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd

GENETIC_CODE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*", "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L", "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q", "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M", "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K", "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V", "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E", "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

CODON_GROUPS = {}
for codon, amino_acid in GENETIC_CODE.items():
    CODON_GROUPS.setdefault(amino_acid, []).append(codon)


def mutate_nucleotide_sequence(
    sequence: str, substitution_rate: float, insertion_rate: float, deletion_rate: float
) -> str:
    nucleotides = ["A", "T", "G", "C"]
    mutated_sequence = list(sequence)
    i = 0
    while i < len(mutated_sequence):
        rand_val = random.random()
        if rand_val < deletion_rate and len(mutated_sequence) > 10:
            del mutated_sequence[i]
            continue
        elif rand_val < deletion_rate + insertion_rate:
            mutated_sequence.insert(i, random.choice(nucleotides))
            i += 1
        elif rand_val < deletion_rate + insertion_rate + substitution_rate:
            current_nucleotide = mutated_sequence[i]
            available_nucleotides = [n for n in nucleotides if n != current_nucleotide]
            if available_nucleotides:
                mutated_sequence[i] = random.choice(available_nucleotides)
        i += 1
    return "".join(mutated_sequence)


def augment_codon_sequence(sequence: str, substitution_rate: float) -> str:
    codons = [sequence[i : i + 3] for i in range(0, len(sequence), 3)]
    augmented_codons = []
    for codon in codons:
        if len(codon) == 3 and codon in GENETIC_CODE:
            amino_acid = GENETIC_CODE[codon]
            synonymous_codons = CODON_GROUPS[amino_acid]
            if len(synonymous_codons) > 1 and random.random() < substitution_rate:
                available_codons = [c for c in synonymous_codons if c != codon]
                augmented_codons.append(random.choice(available_codons))
            else:
                augmented_codons.append(codon)
        else:
            augmented_codons.append(codon)
    return "".join(augmented_codons)


def add_gaussian_noise_to_row(row: np.ndarray, noise_level: float) -> np.ndarray:
    feature_std = np.std(row)
    noise_scale = feature_std * noise_level
    noise = np.random.normal(0, noise_scale, row.shape)
    return row + noise


def run_sequence_augmentation(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.input_file)
    original_rows = df.to_dict("records")
    augmented_rows = []

    for _ in range(args.augmentation_factor):
        for row in original_rows:
            new_row = row.copy()
            new_row["5UTR"] = mutate_nucleotide_sequence(
                str(row["5UTR"]),
                args.utr_substitution_rate,
                args.utr_insertion_rate,
                args.utr_deletion_rate,
            )
            new_row["ORF"] = augment_codon_sequence(
                str(row["ORF"]), args.orf_substitution_rate
            )
            new_row["3UTR"] = mutate_nucleotide_sequence(
                str(row["3UTR"]),
                args.utr_substitution_rate,
                args.utr_insertion_rate,
                args.utr_deletion_rate,
            )
            augmented_rows.append(new_row)

    augmented_df = pd.DataFrame(augmented_rows)
    augmented_df.to_csv(args.output_file, index=False)
    print(f"Successfully generated {len(augmented_df)} augmented sequences.")
    print(f"Original count: {len(df)}, Augmented count: {len(augmented_df)}")


def run_feature_augmentation(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.input_file)
    feature_cols = [c for c in df.columns if c not in args.id_columns]
    augmented_dfs = [df.copy()]

    for _ in range(args.augmentation_factor - 1):
        augmented_df = df.copy()
        feature_matrix = augmented_df[feature_cols].values
        noisy_features = np.array(
            [add_gaussian_noise_to_row(row, args.noise_level) for row in feature_matrix]
        )
        augmented_df[feature_cols] = noisy_features
        augmented_dfs.append(augmented_df)

    final_df = pd.concat(augmented_dfs, ignore_index=True)
    final_df.to_csv(args.output_file, index=False)
    print(f"Successfully generated augmented feature set.")
    print(f"Original shape: {df.shape}, Augmented shape: {final_df.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment sequence or feature datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--random-seed", type=int, default=654, help="Random seed for reproducibility.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Sequence augmentation parser
    seq_parser = subparsers.add_parser("sequences", help="Augment sequence data.")
    seq_parser.add_argument("input_file", type=Path, help="Input CSV file with sequence data.")
    seq_parser.add_argument("output_file", type=Path, help="Output CSV file for augmented data.")
    seq_parser.add_argument("--augmentation-factor", type=int, default=2, help="How many times to augment the dataset.")
    seq_parser.add_argument("--utr-substitution-rate", type=float, default=0.015, help="Substitution rate for UTRs.")
    seq_parser.add_argument("--utr-insertion-rate", type=float, default=0.003, help="Insertion rate for UTRs.")
    seq_parser.add_argument("--utr-deletion-rate", type=float, default=0.002, help="Deletion rate for UTRs.")
    seq_parser.add_argument("--orf-substitution-rate", type=float, default=0.15, help="Synonymous codon substitution rate for ORFs.")
    seq_parser.set_defaults(func=run_sequence_augmentation)

    # Feature augmentation parser
    feat_parser = subparsers.add_parser("features", help="Augment feature data with Gaussian noise.")
    feat_parser.add_argument("input_file", type=Path, help="Input CSV file with feature data.")
    feat_parser.add_argument("output_file", type=Path, help="Output CSV file for augmented features.")
    feat_parser.add_argument("--augmentation-factor", type=int, default=2, help="How many times to augment the dataset.")
    feat_parser.add_argument("--noise-level", type=float, default=0.05, help="Noise level as a fraction of feature standard deviation.")
    feat_parser.add_argument("--id-columns", nargs='+', default=["ENSID", "HALFLIFE"], help="Columns to exclude from noise application.")
    feat_parser.set_defaults(func=run_feature_augmentation)

    try:
        args = parser.parse_args()
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        args.func(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
