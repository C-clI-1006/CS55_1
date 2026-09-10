from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _cosine_matrix(left_embeddings, right_embeddings):
    rows = []
    for left in left_embeddings:
        left_vec = left['embedding']
        row = []
        for right in right_embeddings:
            right_vec = right['embedding']
            row.append(F.cosine_similarity(left_vec, right_vec).item())
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32)


def calculate_storyboard_animatic_similarity(storyboard_embeddings, animatic_embeddings):
    matrix = _cosine_matrix(storyboard_embeddings, animatic_embeddings)
    print('Storyboard → Animatic similarity shape:', matrix.shape)
    return matrix


def calculate_animatic_final_similarity_matrix(animatic_embeddings, final_embeddings):
    matrix = _cosine_matrix(animatic_embeddings, final_embeddings)
    print('Animatic → Final similarity shape:', matrix.shape)
    return matrix


def _monotonic_dp(similarity_matrix):
    sim = similarity_matrix.cpu().numpy()
    n_left, n_right = sim.shape

    dp = np.full((n_left, n_right), -np.inf)
    parent = np.full((n_left, n_right), -1, dtype=int)
    dp[0] = sim[0]

    for i in range(1, n_left):
        for j in range(n_right):
            previous_scores = dp[i - 1, :j + 1]
            best_prev_j = int(np.argmax(previous_scores))
            dp[i, j] = previous_scores[best_prev_j] + sim[i, j]
            parent[i, j] = best_prev_j

    best_path = np.zeros(n_left, dtype=int)
    best_path[-1] = int(np.argmax(dp[-1]))

    for i in range(n_left - 1, 0, -1):
        best_path[i - 1] = parent[i, best_path[i]]

    return sim, best_path


def sequence_aware_storyboard_animatic_matching(similarity_matrix):
    sim, path = _monotonic_dp(similarity_matrix)
    return [
        {
            'storyboard_panel': i + 1,
            'matched_animatic_shot': int(path[i] + 1),
            'similarity': float(sim[i, path[i]]),
        }
        for i in range(len(path))
    ]


def sequence_aware_animatic_final_matching(similarity_matrix):
    sim, path = _monotonic_dp(similarity_matrix)
    return [
        {
            'animatic_shot': i + 1,
            'matched_final_shot': int(path[i] + 1),
            'similarity': float(sim[i, path[i]]),
        }
        for i in range(len(path))
    ]
