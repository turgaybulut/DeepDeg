import argparse
import csv
from enum import Enum
import multiprocessing as mp
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Set, Tuple, Union

from Bio.Seq import Seq
from Bio.SeqUtils import gc_fraction
from PyBioMed import Pydna, Pyprotein
import gffutils
from modlamp.descriptors import GlobalDescriptor
import numpy as np
import pandas as pd
from pyteomics.parser import expasy_rules, icleave
from tqdm import tqdm

CODON_LENGTH = 3
KILOBASE_DIVISOR = 1000
MISSED_CLEAVAGES = 0
DEFAULT_GTF_PATH = "data/Homo_sapiens.GRCh38.100.chr.gtf.gz"
DEFAULT_BATCH_SIZE = 100
DEFAULT_NUM_CORES = mp.cpu_count()


class FeatureType(Enum):
    BASIC = "basic"
    NUCLEIC_ACID = "nucleic_acid"
    PEPTIDE = "peptide"
    MODLAMP = "modlamp"
    ENZYMATIC = "enzymatic"
    ALL = "all"


def load_gtf(gtf_path: str = DEFAULT_GTF_PATH):
    db_path = gtf_path.replace(".gz", "") + ".db"
    if not os.path.exists(db_path):
        gffutils.create_db(
            gtf_path,
            dbfn=db_path,
            force=True,
            keep_order=True,
            disable_infer_transcripts=True,
            disable_infer_genes=True,
        )
    return gffutils.FeatureDB(db_path)


GTF_DB = load_gtf()


def get_basic_features(ensid, utr5_seq, orf_seq, utr3_seq):
    feats = {
        "basic_5utr_len": len(utr5_seq),
        "basic_cds_len": len(orf_seq),
        "basic_3utr_len": len(utr3_seq),
        "basic_5utr_gc": gc_fraction(utr5_seq) if utr5_seq else 0,
        "basic_cds_gc": gc_fraction(orf_seq) if orf_seq else 0,
        "basic_3utr_gc": gc_fraction(utr3_seq) if utr3_seq else 0,
    }

    try:
        tx = max(
            GTF_DB.children(ensid, featuretype="transcript"),
            key=lambda t: sum(
                (e.end or 0) - (e.start or 0) + 1
                for e in GTF_DB.children(t, featuretype="CDS")
                if e.end is not None and e.start is not None
            ),
        )
        exons = list(GTF_DB.children(tx, featuretype="exon"))
        cds = list(GTF_DB.children(tx, featuretype="CDS"))

        valid_exons = sorted(
            [e for e in exons if e.start is not None and e.end is not None],
            key=lambda ex: ex.start or 0,
        )

        feats["basic_intron_len"] = sum(
            ((valid_exons[i + 1].start or 0) - (valid_exons[i].end or 0) - 1)
            for i in range(len(valid_exons) - 1)
            if valid_exons[i + 1].start is not None and valid_exons[i].end is not None
        )
        feats["basic_orf_exon_density"] = (
            (len(cds) - 1) / (len(orf_seq) / KILOBASE_DIVISOR) if len(orf_seq) else 0
        )
    except Exception:
        print(f"Error getting basic features for {ensid}")
        feats["basic_intron_len"] = 0
        feats["basic_orf_exon_density"] = 0

    return feats


def translate_sequence(sequence: str) -> str:
    sequence = str(sequence)
    remainder = len(sequence) % CODON_LENGTH
    if remainder != 0:
        sequence = sequence[:-remainder]

    return str(Seq(sequence).translate())


def get_modlamp_features(peptide_sequence: str) -> Dict[str, Any]:
    desc = GlobalDescriptor(str(peptide_sequence))
    feature_calculators = {
        "modlamp_length": desc.length,
        "modlamp_molecular_weight": desc.calculate_MW,
        "modlamp_charge": desc.calculate_charge,
        "modlamp_charge_density": desc.charge_density,
        "modlamp_isoelectric_point": desc.isoelectric_point,
        "modlamp_instability_index": desc.instability_index,
        "modlamp_aromaticity": desc.aromaticity,
        "modlamp_aliphatic_index": desc.aliphatic_index,
        "modlamp_boman_index": desc.boman_index,
        "modlamp_hydrophobic_ratio": desc.hydrophobic_ratio,
    }

    features = {}
    for name, func in feature_calculators.items():
        func()
        features[name] = desc.descriptor[0][0]
    return features


def get_peptide_features(peptide_sequence: str) -> Dict[str, Any]:
    protein = Pyprotein.PyProtein(str(peptide_sequence))
    features = {}
    features.update(protein.GetAAComp())
    features.update(protein.GetDPComp())
    features.update(protein.GetCTD())
    features.update(protein.GetMoreauBrotoAuto())
    features.update(protein.GetMoranAuto())
    features.update(protein.GetGearyAuto())
    features.update(protein.GetSOCN())
    features.update(protein.GetQSO())
    features.update(protein.GetPAAC())
    features.update(protein.GetAPAAC())
    return features


def get_nucleic_acid_features(sequence: str) -> Dict[str, Any]:
    dna_sequence = str(sequence).replace("U", "T")
    if not dna_sequence:
        return {}
    dna = Pydna.PyDNA(dna_sequence)
    features = {}
    features.update(dna.GetKmer())
    features.update(dna.GetDAC(all_property=True))
    features.update(dna.GetDCC(all_property=True))
    features.update(dna.GetPseDNC(all_property=True))
    features.update(dna.GetPseKNC(all_property=True))
    return features


def get_enzymatic_cleavage_features(peptide_sequence: str) -> Dict[str, Any]:
    peptide_sequence = str(peptide_sequence)
    features: Dict[str, Any] = {}
    all_cleavage_sites = set()
    total_cleaving_enzymes = 0
    total_cleavage_sites = 0

    enzyme_rules = {
        k: v for k, v in expasy_rules.items() if not k.endswith("_exception")
    }

    for enzyme_name, rule in enzyme_rules.items():
        cleavage_events = list(
            icleave(peptide_sequence, rule, missed_cleavages=MISSED_CLEAVAGES)
        )
        if len(cleavage_events) > 1:
            site_indices = {event[0] for event in cleavage_events[1:]}
            site_count = len(site_indices)
            features[f"cleavage_weight_{enzyme_name}"] = site_count
            total_cleavage_sites += site_count
            total_cleaving_enzymes += 1
            all_cleavage_sites.update(site_indices)
        else:
            features[f"cleavage_weight_{enzyme_name}"] = 0

    features["total_cleaving_enzymes"] = total_cleaving_enzymes
    features["total_cleavage_sites"] = total_cleavage_sites

    if len(all_cleavage_sites) < 2:
        features["nearest_cleavage_distance"] = 0
    else:
        sorted_sites = sorted(list(all_cleavage_sites))
        distances = np.diff(sorted_sites)
        features["nearest_cleavage_distance"] = (
            np.min(distances) if len(distances) > 0 else 0
        )
    return features


def _expand_all_features(feature_types: Set[FeatureType]) -> Set[FeatureType]:
    if FeatureType.ALL in feature_types:
        return {ft for ft in FeatureType if ft != FeatureType.ALL}
    return feature_types


def extract_features(row: pd.Series, feature_types: Set[FeatureType]) -> Dict[str, Any]:
    utr5_seq, orf_seq, utr3_seq = str(row["5UTR"]), str(row["ORF"]), str(row["3UTR"])
    expanded_types = _expand_all_features(feature_types)
    features: Dict[str, Any] = {}

    if FeatureType.BASIC in expanded_types:
        features.update(get_basic_features(row["ENSID"], utr5_seq, orf_seq, utr3_seq))

    if FeatureType.NUCLEIC_ACID in expanded_types:
        features.update(
            {f"5UTR_{k}": v for k, v in get_nucleic_acid_features(utr5_seq).items()}
        )
        features.update(
            {f"ORF_{k}": v for k, v in get_nucleic_acid_features(orf_seq).items()}
        )
        features.update(
            {f"3UTR_{k}": v for k, v in get_nucleic_acid_features(utr3_seq).items()}
        )

    peptide_sequence = translate_sequence(orf_seq)
    if peptide_sequence and "*" not in peptide_sequence:
        if FeatureType.PEPTIDE in expanded_types:
            features.update(
                {
                    f"peptide_{k}": v
                    for k, v in get_peptide_features(peptide_sequence).items()
                }
            )
        if FeatureType.MODLAMP in expanded_types:
            features.update(get_modlamp_features(peptide_sequence))
        if FeatureType.ENZYMATIC in expanded_types:
            features.update(get_enzymatic_cleavage_features(peptide_sequence))
    return features


def validate_file_path(file_path: Union[str, Path], must_exist: bool = True) -> Path:
    path = Path(file_path)
    if must_exist and not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if must_exist and not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    return path


def validate_output_directory(output_path: Union[str, Path]) -> Path:
    path = Path(output_path)
    if path.exists() and not path.is_file():
        raise ValueError(f"Output path exists but is not a file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def parse_feature_types(feature_args: List[str]) -> Set[FeatureType]:
    feature_types = set()
    for arg in feature_args:
        try:
            if arg.lower() == "all":
                return {FeatureType.ALL}
            feature_types.add(FeatureType(arg.lower()))
        except ValueError:
            valid_types = [ft.value for ft in FeatureType]
            raise ValueError(f"Invalid feature type '{arg}'. Valid types: {valid_types}")
    return feature_types if feature_types else {FeatureType.ALL}


def process_chunk(chunk_data: tuple) -> List[Dict]:
    df_chunk, feature_types = chunk_data
    return df_chunk.apply(
        lambda row: extract_features(row, feature_types), axis=1
    ).tolist()


def load_existing_processed_data(output_path: Path) -> Set[str]:
    if not output_path.exists():
        return set()
    try:
        processed_df = pd.read_csv(output_path)
        if "ENSID" not in processed_df.columns:
            print("Warning: 'ENSID' column not found. Processing all sequences.")
            return set()
        processed_ensids = set(processed_df["ENSID"])
        print(f"Found {len(processed_ensids)} completed sequences. Resuming...")
        return processed_ensids
    except (pd.errors.EmptyDataError, FileNotFoundError):
        print("Output file is empty or corrupted. Starting from scratch.")
        return set()


def filter_unprocessed_sequences(
    sequence_df: pd.DataFrame, processed_ensids: Set[str]
) -> pd.DataFrame:
    if not processed_ensids:
        return sequence_df
    return sequence_df[~sequence_df["ENSID"].isin(list(processed_ensids))].copy()


def create_chunks(sequence_df: pd.DataFrame, batch_size: int) -> List[pd.DataFrame]:
    num_batches = int(np.ceil(len(sequence_df) / batch_size))
    return [
        sequence_df.iloc[i * batch_size : (i + 1) * batch_size]
        for i in range(num_batches)
    ]


def process_batch_parallel(
    chunk: pd.DataFrame, feature_types: Set[FeatureType], num_cores: int
) -> List[Dict]:
    chunk_splits = np.array_split(chunk, num_cores)
    chunk_data = [(split, feature_types) for split in chunk_splits]
    with mp.Pool(num_cores) as pool:
        feature_list_of_lists = pool.map(process_chunk, chunk_data)
    return [item for sublist in feature_list_of_lists for item in sublist]


def save_batch_results(
    feature_list: List[Dict], chunk: pd.DataFrame, output_path: Path
) -> None:
    if not feature_list:
        return
    features_df = pd.DataFrame(feature_list).fillna(0)
    batch_info_df = chunk[["ENSID", "HALFLIFE"]].reset_index(drop=True)
    final_df = pd.concat([batch_info_df, features_df], axis=1)
    header = not output_path.exists()
    final_df.to_csv(output_path, mode="a", header=header, index=False)


def get_csv_shape_optimized(file_path: Path) -> Tuple[int, int]:
    with open(file_path, "r", newline="") as f:
        reader = csv.reader(f)
        num_columns = len(next(reader))
        num_rows = sum(1 for _ in reader) + 1
    return (num_rows, num_columns)


def generate_and_save_features(
    input_path: Path,
    output_path: Path,
    feature_types: Set[FeatureType],
    batch_size: int,
    num_cores: int,
) -> None:
    print(f"Loading data from {input_path}...")
    sequence_df = pd.read_csv(input_path)
    processed_ensids = load_existing_processed_data(output_path)
    sequence_df = filter_unprocessed_sequences(sequence_df, processed_ensids)

    if sequence_df.empty:
        print("All sequences have already been processed.")
        return

    print(f"Processing {len(sequence_df)} new sequences...")
    print(f"Feature types: {[ft.value for ft in feature_types]}")
    print(f"Using {num_cores} cores for parallel processing.")

    df_chunks = create_chunks(sequence_df, batch_size)
    for chunk in tqdm(df_chunks, desc="Processing Batches"):
        feature_list = process_batch_parallel(chunk, feature_types, num_cores)
        save_batch_results(feature_list, chunk, output_path)

    print("Feature extraction complete.")
    final_shape = get_csv_shape_optimized(output_path)
    print(f"Final feature matrix shape: {final_shape}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract features from sequence data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_file", type=str, help="Path to input CSV file with sequence data"
    )
    parser.add_argument(
        "output_file", type=str, help="Path to output CSV file for features"
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=["all"],
        choices=[ft.value for ft in FeatureType],
        help="Feature types to extract",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--num-cores",
        type=int,
        default=DEFAULT_NUM_CORES,
        help="Number of CPU cores for parallel processing",
    )
    return parser.parse_args()


def main() -> None:
    try:
        args = parse_arguments()
        input_path = validate_file_path(args.input_file)
        output_path = validate_output_directory(args.output_file)
        feature_types = parse_feature_types(args.features)

        if args.num_cores <= 0:
            raise ValueError("Number of cores must be positive")
        if args.batch_size <= 0:
            raise ValueError("Batch size must be positive")

        generate_and_save_features(
            input_path=input_path,
            output_path=output_path,
            feature_types=feature_types,
            batch_size=args.batch_size,
            num_cores=args.num_cores,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
