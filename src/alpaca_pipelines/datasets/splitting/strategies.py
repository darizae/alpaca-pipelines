from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable

from alpaca_pipelines.datasets.contracts import SnippetEntry

SplitResult = dict[str, list[SnippetEntry]]
SplitFunction = Callable[[list[SnippetEntry], int, tuple[float, float, float]], SplitResult]


def _split_list(
    items: list[SnippetEntry],
    fractions: tuple[float, float, float],
) -> SplitResult:
    n = len(items)
    boundary_train = int(fractions[0] * n)
    boundary_val = boundary_train + int(fractions[1] * n)
    return {
        "train": items[:boundary_train],
        "val": items[boundary_train:boundary_val],
        "test": items[boundary_val:],
    }


def split_random(
    snippets: list[SnippetEntry],
    seed: int,
    fractions: tuple[float, float, float],
) -> SplitResult:
    rng = random.Random(seed)
    shuffled = list(snippets)
    rng.shuffle(shuffled)
    return _split_list(shuffled, fractions)


def split_quality_balanced(
    snippets: list[SnippetEntry],
    seed: int,
    fractions: tuple[float, float, float],
) -> SplitResult:
    rng = random.Random(seed)
    quality_buckets: dict[int | None, list[SnippetEntry]] = defaultdict(list)
    for snippet in snippets:
        quality_buckets[snippet.quality].append(snippet)

    merged: SplitResult = {"train": [], "val": [], "test": []}
    for _quality, bucket in quality_buckets.items():
        sub_seed = rng.randint(0, 1 << 30)
        bucket_split = split_random(bucket, sub_seed, fractions)
        for split_name in merged:
            merged[split_name].extend(bucket_split[split_name])

    for split_name in merged:
        rng.shuffle(merged[split_name])
    return merged


def _separate_by_session_key(
    snippets: list[SnippetEntry],
) -> tuple[dict[str, list[SnippetEntry]], list[SnippetEntry]]:
    session_buckets: dict[str, list[SnippetEntry]] = defaultdict(list)
    ungrouped: list[SnippetEntry] = []
    for snippet in snippets:
        if snippet.session_key is None:
            ungrouped.append(snippet)
        else:
            session_buckets[snippet.session_key].append(snippet)
    return session_buckets, ungrouped


def _assign_keys_to_splits(
    keys: list[str],
    fractions: tuple[float, float, float],
) -> dict[str, str]:
    n_keys = len(keys)
    boundary_train = int(fractions[0] * n_keys)
    boundary_val = boundary_train + int(fractions[1] * n_keys)

    key_to_split: dict[str, str] = {}
    for idx, key in enumerate(keys):
        if idx < boundary_train:
            key_to_split[key] = "train"
        elif idx < boundary_val:
            key_to_split[key] = "val"
        else:
            key_to_split[key] = "test"
    return key_to_split


def split_clipwise_balanced(
    snippets: list[SnippetEntry],
    seed: int,
    fractions: tuple[float, float, float],
) -> SplitResult:
    rng = random.Random(seed)
    session_buckets, ungrouped = _separate_by_session_key(snippets)

    session_keys = list(session_buckets.keys())
    rng.shuffle(session_keys)

    key_to_split = _assign_keys_to_splits(session_keys, fractions)

    merged: SplitResult = {"train": [], "val": [], "test": []}
    for session_key, bucket in session_buckets.items():
        split_name = key_to_split[session_key]
        merged[split_name].extend(bucket)

    if ungrouped:
        ungrouped_split = split_random(ungrouped, rng.randint(0, 1 << 30), fractions)
        for split_name in merged:
            merged[split_name].extend(ungrouped_split[split_name])

    for split_name in merged:
        rng.shuffle(merged[split_name])
    return merged


def split_quality_and_clipwise_balanced(
    snippets: list[SnippetEntry],
    seed: int,
    fractions: tuple[float, float, float],
) -> SplitResult:
    """Assign whole sessions atomically to splits (no session leakage).

    Session-keyed snippets are assigned to splits by shuffled session order.
    Ungrouped snippets (session_key=None, e.g. mined noise) are split via
    quality_balanced to distribute quality levels across splits.

    NOTE: This does NOT stratify session assignment by quality distribution.
    Session assignment is random. For true quality-stratified session
    assignment, a greedy/knapsack heuristic would be needed.
    """
    rng = random.Random(seed)
    session_buckets, ungrouped = _separate_by_session_key(snippets)

    session_keys = list(session_buckets.keys())
    rng.shuffle(session_keys)

    key_to_split = _assign_keys_to_splits(session_keys, fractions)

    merged: SplitResult = {"train": [], "val": [], "test": []}
    for session_key, bucket in session_buckets.items():
        split_name = key_to_split[session_key]
        merged[split_name].extend(bucket)

    if ungrouped:
        ungrouped_split = split_quality_balanced(ungrouped, rng.randint(0, 1 << 30), fractions)
        for split_name in merged:
            merged[split_name].extend(ungrouped_split[split_name])

    for split_name in merged:
        rng.shuffle(merged[split_name])
    return merged


STRATEGY_FUNCTIONS: dict[str, SplitFunction] = {
    "random": split_random,
    "quality_balanced": split_quality_balanced,
    "clipwise_balanced": split_clipwise_balanced,
    "quality_and_clipwise_balanced": split_quality_and_clipwise_balanced,
}


def apply_split(
    snippets: list[SnippetEntry],
    strategy_name: str,
    seed: int,
    fractions: tuple[float, float, float],
) -> SplitResult:
    if strategy_name not in STRATEGY_FUNCTIONS:
        raise ValueError(
            f"Unknown split strategy '{strategy_name}'. "
            f"Available: {sorted(STRATEGY_FUNCTIONS.keys())}"
        )
    split_fn = STRATEGY_FUNCTIONS[strategy_name]
    return split_fn(snippets, seed, fractions)
