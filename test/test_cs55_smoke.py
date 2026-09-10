import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
sys.path.insert(0, str(SRC))

from cs55_demo import load_config
from cs55_demo.pipeline_runner import run_full_pipeline


class DummyEmbedder:
    """Deterministic lightweight embedder for packaging tests only."""
    def embed(self, image_path):
        # Use file bytes to create a deterministic 8-D vector.
        data = Path(image_path).read_bytes()
        vals = [sum(data[i::8]) % 997 for i in range(8)]
        v = torch.tensor(vals, dtype=torch.float32).reshape(1, -1)
        return torch.nn.functional.normalize(v, p=2, dim=-1)


def test_smoke():
    config = load_config(ROOT / 'config.sample.json')
    result, bundle = run_full_pipeline(
        config,
        use_cache=False,
        embedder=DummyEmbedder(),
    )
    assert result['dataset_name'] == 'SampleDemo'
    assert result['evidence_chain']['artefact_count'] == 6
    assert result['evidence_chain']['relationship_count'] == 5
    assert 0 <= result['final_score']['integrity'] <= 1
    assert result['final_score']['completeness'] == 1.0
    assert bundle['hash_method'] == 'sha256'


if __name__ == '__main__':
    test_smoke()
    print('CS55 smoke test passed')
