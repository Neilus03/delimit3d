from __future__ import annotations

import csv
import os
from functools import lru_cache
from pathlib import Path

SCANNET_EVAL_BENCHMARK_ALL = "all"
SCANNET_EVAL_BENCHMARK_20 = "scannet20"
SCANNET_EVAL_BENCHMARK_200 = "scannet200"

DEFAULT_SCANNET_EVAL_BENCHMARK = SCANNET_EVAL_BENCHMARK_200
DEFAULT_SCANNET_EVAL_BENCHMARKS = (
    SCANNET_EVAL_BENCHMARK_20,
    SCANNET_EVAL_BENCHMARK_200,
)

VALID_CLASS_IDS_20 = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    14,
    16,
    24,
    28,
    33,
    34,
    36,
    39,
)

VALID_CLASS_IDS_200 = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    21,
    22,
    23,
    24,
    26,
    27,
    28,
    29,
    31,
    32,
    33,
    34,
    35,
    36,
    38,
    39,
    40,
    41,
    42,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    54,
    55,
    56,
    57,
    58,
    59,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    82,
    84,
    86,
    87,
    88,
    89,
    90,
    93,
    95,
    96,
    97,
    98,
    99,
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    110,
    112,
    115,
    116,
    118,
    120,
    121,
    122,
    125,
    128,
    130,
    131,
    132,
    134,
    136,
    138,
    139,
    140,
    141,
    145,
    148,
    154,
    155,
    156,
    157,
    159,
    161,
    163,
    165,
    166,
    168,
    169,
    170,
    177,
    180,
    185,
    188,
    191,
    193,
    195,
    202,
    208,
    213,
    214,
    221,
    229,
    230,
    232,
    233,
    242,
    250,
    261,
    264,
    276,
    283,
    286,
    300,
    304,
    312,
    323,
    325,
    331,
    342,
    356,
    370,
    392,
    395,
    399,
    408,
    417,
    488,
    540,
    562,
    570,
    572,
    581,
    609,
    748,
    776,
    1156,
    1163,
    1164,
    1165,
    1166,
    1167,
    1168,
    1169,
    1170,
    1171,
    1172,
    1173,
    1174,
    1175,
    1176,
    1178,
    1179,
    1180,
    1181,
    1182,
    1183,
    1184,
    1185,
    1186,
    1187,
    1188,
    1189,
    1190,
    1191,
)

SUPPORTED_SCANNET_EVAL_BENCHMARKS = frozenset(
    {
        SCANNET_EVAL_BENCHMARK_ALL,
        SCANNET_EVAL_BENCHMARK_20,
        SCANNET_EVAL_BENCHMARK_200,
    }
)


def normalize_scannet_eval_benchmark(benchmark: str | None) -> str:
    value = (
        benchmark
        or os.environ.get("DELIMIT3D_SCANNET_EVAL_BENCHMARK")
        or os.environ.get("CHORUS_SCANNET_EVAL_BENCHMARK")
        or DEFAULT_SCANNET_EVAL_BENCHMARK
    )
    normalized = str(value).strip().lower()

    aliases = {
        "20": SCANNET_EVAL_BENCHMARK_20,
        "200": SCANNET_EVAL_BENCHMARK_200,
        "scannetv2_20": SCANNET_EVAL_BENCHMARK_20,
        "scannetv2_200": SCANNET_EVAL_BENCHMARK_200,
    }
    normalized = aliases.get(normalized, normalized)

    if normalized not in SUPPORTED_SCANNET_EVAL_BENCHMARKS:
        supported = ", ".join(sorted(SUPPORTED_SCANNET_EVAL_BENCHMARKS))
        raise ValueError(
            f"Unsupported ScanNet eval benchmark '{benchmark}'. Expected one of: {supported}."
        )

    return normalized


def parse_scannet_eval_benchmarks(benchmarks: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if benchmarks is None:
        return list(DEFAULT_SCANNET_EVAL_BENCHMARKS)

    if isinstance(benchmarks, str):
        raw_values = [part.strip() for part in benchmarks.split(",") if part.strip()]
    else:
        raw_values = [str(part).strip() for part in benchmarks if str(part).strip()]

    if not raw_values:
        raw_values = list(DEFAULT_SCANNET_EVAL_BENCHMARKS)

    normalized_values: list[str] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        normalized = normalize_scannet_eval_benchmark(raw_value)
        if normalized in seen:
            continue
        seen.add(normalized)
        normalized_values.append(normalized)

    return normalized_values


def primary_scannet_eval_benchmark(benchmarks: str | list[str] | tuple[str, ...] | None) -> str:
    parsed = parse_scannet_eval_benchmarks(benchmarks)
    if SCANNET_EVAL_BENCHMARK_20 in parsed:
        return SCANNET_EVAL_BENCHMARK_20
    return parsed[0]


def get_valid_class_ids_for_benchmark(benchmark: str) -> set[int] | None:
    normalized = normalize_scannet_eval_benchmark(benchmark)
    if normalized == SCANNET_EVAL_BENCHMARK_ALL:
        return None
    if normalized == SCANNET_EVAL_BENCHMARK_20:
        return set(VALID_CLASS_IDS_20)
    return set(VALID_CLASS_IDS_200)


def _candidate_scannet_metadata_roots() -> list[Path]:
    candidates: list[Path] = []

    env_root = os.environ.get(
        "DELIMIT3D_SCANNET_METADATA_ROOT",
        os.environ.get("CHORUS_SCANNET_METADATA_ROOT"),
    )
    if env_root:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(env_root))).resolve())

    litept_root = os.environ.get("LITEPT_ROOT")
    if litept_root:
        candidates.append(
            (
                Path(os.path.expanduser(os.path.expandvars(litept_root)))
                / "datasets"
                / "preprocessing"
                / "scannet"
                / "meta_data"
            ).resolve()
        )

    # --- Search the current working directory ---
    candidates.append(Path.cwd().resolve())

    this_file = Path(__file__).resolve()
    candidates.append((this_file.parent / "meta_data").resolve())

    for parent in this_file.parents:
        candidates.append((parent / "LitePT" / "datasets" / "preprocessing" / "scannet" / "meta_data").resolve())
        # --- Also check parent directories directly ---
        candidates.append(parent.resolve())

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)

    return unique_candidates


def _has_readable_scannet_labels(candidate: Path) -> bool:
    labels_path = candidate / "scannetv2-labels.combined.tsv"
    try:
        return labels_path.is_file() and os.access(labels_path, os.R_OK)
    except OSError:
        return False


def resolve_scannet_metadata_root() -> Path:
    for candidate in _candidate_scannet_metadata_roots():
        if _has_readable_scannet_labels(candidate):
            return candidate

    searched = "\n".join(f"  - {candidate}" for candidate in _candidate_scannet_metadata_roots())
    raise FileNotFoundError(
        "Could not locate a READABLE ScanNet metadata root with 'scannetv2-labels.combined.tsv'. "
        "Set DELIMIT3D_SCANNET_METADATA_ROOT to the directory containing the official ScanNet metadata files.\n"
        f"Searched:\n{searched}"
    )


@lru_cache(maxsize=1)
def load_raw_category_label_map() -> dict[str, dict[str, int]]:
    labels_path = resolve_scannet_metadata_root() / "scannetv2-labels.combined.tsv"
    label_map: dict[str, dict[str, int]] = {}

    with labels_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            raw_category = str(row.get("raw_category", "")).strip().lower()
            if not raw_category or raw_category in label_map:
                continue

            label_map[raw_category] = {
                "id": int(row["id"]),
                "nyu40id": int(row["nyu40id"]),
            }

    return label_map
