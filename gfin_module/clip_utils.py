"""
CLIP utilities for GFIN improvements.

Provides:
- Text-based classifier weight initialization using CLIP text encoder
- Visual feature extraction using CLIP image encoder
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

CLIP_DIMS = {
    "ViT-B-32": 512,
    "ViT-B-16": 512,
    "ViT-L-14": 768,
}

VERB_TEMPLATES = [
    "a photo of a person {} a {}",
    "a person {} a {}",
    "someone {} a {}",
]


def _load_clip_model(model_name="ViT-B-32", pretrained="openai"):
    try:
        import open_clip
    except ImportError:
        raise ImportError(
            "open_clip_torch is required for CLIP features. "
            "Install: pip install open_clip_torch"
        )
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, preprocess, tokenizer


def _clean_verb_name(verb):
    """Clean verb/action name for text template generation.

    Handles V-COCO format ('hold obj', 'talk_on_phone instr') by
    stripping role suffixes and replacing underscores with spaces.
    """
    parts = verb.split()
    if parts[-1] in ("obj", "instr"):
        return " ".join(parts[:-1]).replace("_", " ")
    return verb.replace("_", " ")


@torch.no_grad()
def build_clip_text_classifier(
    verbs,
    repr_dim,
    objects=None,
    clip_model_name="ViT-B-32",
):
    """
    Build classifier weights from CLIP text embeddings of verb descriptions.

    For each verb, encode multiple text templates (optionally averaged over
    objects) to get a robust text representation, then project to repr_dim.

    Args:
        verbs: list of verb/action name strings, length = num_classes
        repr_dim: target dimension for classifier weights
        objects: optional list of object name strings for template diversity
        clip_model_name: CLIP model variant

    Returns:
        Tensor of shape (num_verbs, repr_dim) - classifier weight init
    """
    model, _, tokenizer = _load_clip_model(clip_model_name)
    clip_dim = CLIP_DIMS.get(clip_model_name, 512)
    device = next(model.parameters()).device

    all_verb_feats = []
    obj_names = objects if objects else ["object"]

    for verb in verbs:
        verb_str = _clean_verb_name(verb)
        template_feats = []
        for template in VERB_TEMPLATES:
            texts = [template.format(verb_str, obj) for obj in obj_names]
            tokens = tokenizer(texts).to(device)
            feats = model.encode_text(tokens)
            feats = F.normalize(feats.float(), dim=-1)
            template_feats.append(feats.mean(dim=0))
        verb_feat = torch.stack(template_feats).mean(dim=0)
        verb_feat = F.normalize(verb_feat, dim=-1)
        all_verb_feats.append(verb_feat)

    text_feats = torch.stack(all_verb_feats)  # (num_verbs, clip_dim)

    if clip_dim != repr_dim:
        proj = nn.Linear(clip_dim, repr_dim, bias=False)
        nn.init.xavier_uniform_(proj.weight)
        proj = proj.to(device)
        text_feats = proj(text_feats)

    logger.info(
        "Built CLIP text classifier: %d verbs, clip_dim=%d, repr_dim=%d",
        len(verbs), clip_dim, repr_dim,
    )
    return text_feats.detach().cpu()


class CLIPVisualEncoder(nn.Module):
    """
    Frozen CLIP visual encoder that extracts global image features.
    Projects CLIP features to target hidden dimension for fusion with GCE.
    """

    def __init__(self, hidden_dim=256, clip_model_name="ViT-B-32"):
        super().__init__()
        self.clip_model, self.preprocess, _ = _load_clip_model(clip_model_name)
        clip_dim = CLIP_DIMS.get(clip_model_name, 512)
        self.proj = nn.Linear(clip_dim, hidden_dim)
        self.hidden_dim = hidden_dim

        for p in self.clip_model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode(self, images):
        """
        Extract CLIP visual features from images.

        Args:
            images: list of Tensor (C, H, W) with potentially different sizes,
                    or Tensor (B, C, H, W) with uniform size.

        Returns:
            Tensor (B, clip_dim) - global image features
        """
        if isinstance(images, (list, tuple)):
            resized = []
            for img in images:
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                r = F.interpolate(
                    img, size=(224, 224), mode="bilinear", align_corners=False
                )
                resized.append(r.squeeze(0))
            batch = torch.stack(resized)
        else:
            batch = F.interpolate(
                images, size=(224, 224), mode="bilinear", align_corners=False
            )

        feats = self.clip_model.encode_image(batch)
        return F.normalize(feats.float(), dim=-1)

    def forward(self, images):
        """Extract and project CLIP features.

        Returns:
            Tensor (B, hidden_dim) - projected features
        """
        feats = self.encode(images)
        return self.proj(feats)
