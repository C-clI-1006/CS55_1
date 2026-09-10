from __future__ import annotations

import hashlib


def calculate_sha256(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def build_storyboard_artefacts(storyboard_files):
    artefacts = []
    seen_hashes = set()

    for file_path in storyboard_files:
        file_hash = calculate_sha256(file_path)
        if file_hash in seen_hashes:
            continue
        seen_hashes.add(file_hash)
        artefacts.append({
            'artefact_hash': file_hash,
            'artefact_type': 'image',
            'evidence': [],
            'attributes': {},
        })

    return artefacts


def build_libevchain_bundle(config, storyboard_files):
    storyboard_artefacts = build_storyboard_artefacts(storyboard_files)
    animatic_hash = calculate_sha256(config['animatic_path'])
    final_hash = calculate_sha256(config['final_path'])

    animatic_artefact = {
        'artefact_hash': animatic_hash,
        'artefact_type': 'video',
        'evidence': [
            {
                'hash': item['artefact_hash'],
                'relationship_type': 'storyboard-to-animation',
                'attributes': {},
            }
            for item in storyboard_artefacts
        ],
        'attributes': {},
    }

    final_artefact = {
        'artefact_hash': final_hash,
        'artefact_type': 'video',
        'evidence': [
            {
                'hash': animatic_hash,
                'relationship_type': 'animation-to-final',
                'attributes': {},
            }
        ],
        'attributes': {},
    }

    return {
        'hash_method': 'sha256',
        'final_artefact_hash': final_hash,
        'artefacts': storyboard_artefacts + [animatic_artefact, final_artefact],
    }
