from __future__ import annotations

import numpy as np

from libevchain.evidence_chain import EvidenceChain
from libevchain.score import Score
from libevchain.pipeline import EvidencePass, Pipeline
from libevchain.types.artefact_types import ArtefactType, register_artefact_type, get_artefact_type
from libevchain.types.relationship_types import RelationshipType, register_relationship_type, get_relationship_type

from .evidence_chain_builder import calculate_sha256


_storyboard_sequence_match_by_hash = {}
_af_sequence_mean_similarity = 0.0
_af_matched_shot_count = 0


def _safe_get_artefact_type(name):
    try:
        return get_artefact_type(name)
    except Exception:
        return None


def _safe_get_relationship_type(name):
    try:
        return get_relationship_type(name)
    except Exception:
        return None


def ensure_cs55_types():
    if _safe_get_artefact_type('image') is None:
        @register_artefact_type('image')
        class ImageType(ArtefactType):
            pass

    if _safe_get_artefact_type('video') is None:
        @register_artefact_type('video')
        class VideoType(ArtefactType):
            pass

    if _safe_get_relationship_type('storyboard-to-animation') is None:
        @register_relationship_type('storyboard-to-animation')
        class StoryboardToAnimation(RelationshipType):
            @staticmethod
            def attributes():
                return {}

    if _safe_get_relationship_type('animation-to-final') is None:
        @register_relationship_type('animation-to-final')
        class AnimationToFinal(RelationshipType):
            @staticmethod
            def attributes():
                return {}


def configure_pass_context(prepared_data):
    global _storyboard_sequence_match_by_hash
    global _af_sequence_mean_similarity
    global _af_matched_shot_count

    mapping = {}
    for file_path, match in zip(
        prepared_data['storyboard_files'],
        prepared_data['ordered_sa_matches'],
    ):
        file_hash = calculate_sha256(file_path)
        if file_hash not in mapping:
            mapping[file_hash] = match

    ordered_af = prepared_data['ordered_af_matches']
    _storyboard_sequence_match_by_hash = mapping
    _af_sequence_mean_similarity = float(
        np.mean([float(x['similarity']) for x in ordered_af])
    )
    _af_matched_shot_count = len(ordered_af)


class StoryboardToAnimationCLIPPass(EvidencePass):
    @staticmethod
    def evaluate(evidence):
        parent = evidence.get_evidence_artefact()
        match = _storyboard_sequence_match_by_hash[parent.artefact_hash]
        similarity = float(match['similarity'])
        return (
            Score(integrity=similarity),
            {
                'matched_animatic_shot': int(match['matched_animatic_shot']),
                'similarity': similarity,
            },
            [
                f"Matched animatic shot: {match['matched_animatic_shot']}",
                f'Sequence-aware CLIP similarity: {similarity:.4f}',
            ],
        )


class AnimationToFinalCLIPPass(EvidencePass):
    @staticmethod
    def evaluate(evidence):
        similarity = float(_af_sequence_mean_similarity)
        return (
            Score(integrity=similarity),
            {
                'matching_method': 'sequence-aware',
                'matched_shots': _af_matched_shot_count,
                'mean_similarity': similarity,
            },
            [
                'Animatic → Final evaluated using sequence-aware shot matching.',
                f'Matched animatic shots: {_af_matched_shot_count}',
                f'Mean CLIP similarity: {similarity:.4f}',
            ],
        )


def build_evidence_chain(bundle):
    ensure_cs55_types()
    return EvidenceChain.from_dict(bundle)


def build_pipeline(metrics):
    ensure_cs55_types()

    def score_merger(final_artefact):
        return Score(
            integrity=float(metrics['coherence']),
            completeness=float(metrics['completeness']),
        )

    return Pipeline(
        artefact_passes={},
        evidence_passes={
            get_relationship_type('storyboard-to-animation'): [StoryboardToAnimationCLIPPass],
            get_relationship_type('animation-to-final'): [AnimationToFinalCLIPPass],
        },
        score_merger=score_merger,
    )
