from __future__ import annotations

from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image


class CLIPEmbedder:
    def __init__(self, model_name='openai/clip-vit-base-patch32', device=None):
        # Lazy import keeps non-CLIP unit tests lightweight.
        from transformers import CLIPModel, CLIPProcessor

        if device is None:
            device = 'mps' if torch.backends.mps.is_available() else 'cpu'

        self.device = device
        self.model_name = model_name
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def embed(self, image_path):
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors='pt')
        pixel_values = inputs['pixel_values'].to(self.device)

        with torch.no_grad():
            outputs = self.model.vision_model(pixel_values=pixel_values)
            pooled = outputs.pooler_output
            features = self.model.visual_projection(pooled)

        return F.normalize(features, p=2, dim=-1).cpu()


def load_processed_files(config):
    storyboard_files = sorted(Path(config['storyboard_processed_dir']).rglob('*.png'))
    animatic_files = sorted(Path(config['animatic_processed_dir']).rglob('*.png'))
    final_files = sorted(Path(config['final_processed_dir']).rglob('*.png'))

    if not storyboard_files:
        raise ValueError(f"No storyboard PNG files found in {config['storyboard_processed_dir']}")
    if not animatic_files:
        raise ValueError(f"No animatic PNG files found in {config['animatic_processed_dir']}")
    if not final_files:
        raise ValueError(f"No final PNG files found in {config['final_processed_dir']}")

    return storyboard_files, animatic_files, final_files


def generate_all_embeddings(config, embedder=None):
    if embedder is None:
        embedder = CLIPEmbedder(
            model_name=config.get('clip_model', 'openai/clip-vit-base-patch32')
        )

    storyboard_files, animatic_files, final_files = load_processed_files(config)

    storyboard_embeddings = [
        {
            'panel_index': i,
            'panel_path': str(path),
            'embedding': embedder.embed(path),
        }
        for i, path in enumerate(storyboard_files, start=1)
    ]

    animatic_embeddings = [
        {
            'shot_id': i,
            'keyframe_path': str(path),
            'embedding': embedder.embed(path),
        }
        for i, path in enumerate(animatic_files, start=1)
    ]

    final_embeddings = [
        {
            'shot_id': i,
            'keyframe_path': str(path),
            'embedding': embedder.embed(path),
        }
        for i, path in enumerate(final_files, start=1)
    ]

    return {
        'storyboard_files': storyboard_files,
        'storyboard_embeddings': storyboard_embeddings,
        'animatic_embeddings': animatic_embeddings,
        'final_embeddings': final_embeddings,
    }


def save_embedding_cache(data, cache_path):
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert Path values to strings for broad torch serialization compatibility.
    serializable = dict(data)
    serializable['storyboard_files'] = [str(p) for p in data['storyboard_files']]
    torch.save(serializable, cache_path)


def load_embedding_cache(cache_path):
    cache_path = Path(cache_path)
    try:
        data = torch.load(cache_path, map_location='cpu', weights_only=False)
    except TypeError:
        data = torch.load(cache_path, map_location='cpu')
    data['storyboard_files'] = [Path(p) for p in data['storyboard_files']]
    return data


def prepare_embeddings(config, use_cache=True, cache_path=None, embedder=None):
    if cache_path is None:
        cache_path = Path('cache') / f"{config['dataset_name']}_embeddings.pt"

    cache_path = Path(cache_path)
    if use_cache and cache_path.exists():
        print(f'Using cached embeddings: {cache_path}')
        return load_embedding_cache(cache_path)

    print('Generating CLIP embeddings...')
    data = generate_all_embeddings(config, embedder=embedder)

    if use_cache:
        save_embedding_cache(data, cache_path)
        print(f'Embedding cache saved: {cache_path}')

    return data
