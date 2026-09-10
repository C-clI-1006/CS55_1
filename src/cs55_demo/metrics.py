from __future__ import annotations

import numpy as np


def build_chain_rows(prepared_data):
    """Build Storyboard -> Animatic -> Final chain rows exactly as in the research notebook."""
    ordered_sa = prepared_data['ordered_sa_matches']
    ordered_af = prepared_data['ordered_af_matches']

    af_by_animatic = {
        int(item['animatic_shot']): item
        for item in ordered_af
    }

    rows = []
    for sa in ordered_sa:
        animatic_shot = int(sa['matched_animatic_shot'])
        af = af_by_animatic.get(animatic_shot)

        if af is None:
            final_shot = None
            af_similarity = np.nan
            coherence = np.nan
            confidence = np.nan
        else:
            final_shot = int(af['matched_final_shot'])
            af_similarity = float(af['similarity'])
            sa_similarity = float(sa['similarity'])
            coherence = float(np.mean([sa_similarity, af_similarity]))
            confidence = float(max(0.0, min(1.0, 1.0 - abs(sa_similarity - af_similarity))))

        rows.append({
            'storyboard_panel': int(sa['storyboard_panel']),
            'animatic_shot': animatic_shot,
            'final_shot': final_shot,
            'storyboard_animatic_similarity': float(sa['similarity']),
            'animatic_final_similarity': af_similarity,
            'chain_coherence_measurement': coherence,
            'confidence_measurement': confidence,
        })

    return rows


def calculate_chain_metrics(prepared_data, expected_final_count=None):
    """
    Reproduce the metric definitions used in CS55_CrossModal_Coherence-step 2:

    - Chain Coherence: average of per-chain two-stage similarities.
    - Confidence: average of 1 - absolute two-stage similarity gap per chain.
    - Evidence Coverage: mean of used-animatic coverage and used-final coverage.
    - Completeness: final shots used by the chain score 1.0; unused final shots score 0.5.
    - Stage similarities: mean sequence-aware similarity for each relationship stage.
    """
    ordered_sa = prepared_data['ordered_sa_matches']
    ordered_af = prepared_data['ordered_af_matches']
    animatic_embeddings = prepared_data['animatic_embeddings']
    final_embeddings = prepared_data['final_embeddings']

    chain_rows = build_chain_rows(prepared_data)

    coherence_values = [
        row['chain_coherence_measurement']
        for row in chain_rows
        if not np.isnan(row['chain_coherence_measurement'])
    ]
    confidence_values = [
        row['confidence_measurement']
        for row in chain_rows
        if not np.isnan(row['confidence_measurement'])
    ]

    coherence = float(np.mean(coherence_values)) if coherence_values else 0.0
    confidence = float(np.mean(confidence_values)) if confidence_values else 0.0

    used_animatic = {
        int(row['animatic_shot'])
        for row in chain_rows
        if row['animatic_shot'] is not None
    }
    used_final = {
        int(row['final_shot'])
        for row in chain_rows
        if row['final_shot'] is not None
    }

    total_animatic = len(animatic_embeddings)
    total_final_available = len(final_embeddings)

    animatic_coverage = (
        len(used_animatic) / total_animatic
        if total_animatic else 0.0
    )
    final_coverage = (
        len(used_final) / total_final_available
        if total_final_available else 0.0
    )
    coverage = float(np.mean([animatic_coverage, final_coverage]))

    if expected_final_count is None:
        expected_final_count = total_final_available
    expected_final_count = int(expected_final_count)

    # Notebook rule: an evidenced final shot is 'available' (1.0),
    # otherwise it is 'partial' (0.5). No unavailable shots are assigned here.
    final_status = []
    for final_shot in range(1, expected_final_count + 1):
        if final_shot in used_final:
            status = 'available'
            score = 1.0
        else:
            status = 'partial'
            score = 0.5
        final_status.append({
            'final_shot': final_shot,
            'evidence_status': status,
            'completeness_measurement': score,
        })

    completeness = float(np.mean([
        item['completeness_measurement']
        for item in final_status
    ])) if final_status else 0.0

    sa_mean = float(np.mean([
        float(item['similarity'])
        for item in ordered_sa
    ])) if ordered_sa else 0.0

    af_mean = float(np.mean([
        float(item['similarity'])
        for item in ordered_af
    ])) if ordered_af else 0.0

    return {
        'coherence': coherence,
        'confidence': confidence,
        'coverage': coverage,
        'completeness': completeness,
        'storyboard_animatic_similarity': sa_mean,
        'animatic_final_similarity': af_mean,
        'animatic_coverage': float(animatic_coverage),
        'final_coverage': float(final_coverage),
        'used_animatic_shots': len(used_animatic),
        'total_animatic_shots': total_animatic,
        'used_final_shots': len(used_final),
        'total_final_shots': total_final_available,
        'expected_final_count': expected_final_count,
        'chain_rows': chain_rows,
        'final_status': final_status,
    }
