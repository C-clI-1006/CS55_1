from __future__ import annotations

import numpy as np


def calculate_completeness(actual_final_count, expected_final_count):
    if expected_final_count == 0:
        return 0.0
    return float(max(0.0, min(1.0, actual_final_count / expected_final_count)))


def calculate_chain_metrics(prepared_data, expected_final_count=None):
    ordered_sa = prepared_data['ordered_sa_matches']
    ordered_af = prepared_data['ordered_af_matches']
    animatic_embeddings = prepared_data['animatic_embeddings']
    final_embeddings = prepared_data['final_embeddings']

    sa_mean = float(np.mean([float(x['similarity']) for x in ordered_sa]))
    af_mean = float(np.mean([float(x['similarity']) for x in ordered_af]))
    coherence = float(np.mean([sa_mean, af_mean]))
    confidence = float(max(0.0, min(1.0, 1.0 - abs(sa_mean - af_mean))))

    used_animatic = {int(x['matched_animatic_shot']) for x in ordered_sa}
    sa_coverage = len(used_animatic) / len(animatic_embeddings)
    af_coverage = len(ordered_af) / len(animatic_embeddings)
    coverage = float(np.mean([sa_coverage, af_coverage]))

    if expected_final_count is None:
        expected_final_count = len(final_embeddings)

    completeness = calculate_completeness(
        len(final_embeddings), expected_final_count
    )

    return {
        'coherence': coherence,
        'confidence': confidence,
        'coverage': coverage,
        'completeness': completeness,
        'storyboard_animatic_similarity': sa_mean,
        'animatic_final_similarity': af_mean,
    }
