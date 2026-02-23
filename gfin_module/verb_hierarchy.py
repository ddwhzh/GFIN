"""Coarse verb taxonomy for hierarchical classification loss."""

import torch
from typing import List, Tuple

_COARSE_GROUPS = [
    # 0: contact - direct physical interaction
    {'hold', 'carry', 'hug', 'kiss', 'pet', 'scratch', 'lick', 'squeeze',
     'catch', 'throw', 'kick', 'hit', 'push', 'pull', 'drag', 'lift',
     'pick_up', 'pick', 'stab', 'swing', 'block', 'tag', 'grab',
     'touch', 'press', 'shake'},
    # 1: tool / instrument use
    {'cut_with', 'brush_with', 'drink_with', 'eat_with', 'wield', 'hose',
     'stir', 'pour', 'paint', 'grind', 'lasso', 'tie', 'zip', 'stick',
     'light', 'sign', 'type_on', 'text_on', 'talk_on', 'type',
     'work_on_computer'},
    # 2: position / locomotion
    {'ride', 'walk', 'run', 'sit_on', 'sit_at', 'sit', 'lie_on', 'lay',
     'stand_on', 'stand_under', 'stand', 'hop_on', 'jump', 'board',
     'straddle', 'slide', 'exit', 'fly', 'row', 'sail', 'race',
     'drive', 'park', 'stop', 'ski', 'surf', 'skateboard', 'snowboard'},
    # 3: manipulation / state change
    {'open', 'close', 'turn', 'repair', 'install', 'assemble', 'adjust',
     'operate', 'make', 'cook', 'load', 'pack', 'fill', 'set', 'peel',
     'break', 'launch', 'clean', 'wash', 'dry', 'feed', 'groom',
     'shear', 'milk', 'move', 'release', 'lose', 'herd', 'hunt',
     'train', 'control', 'direct', 'cut'},
    # 4: observation / consumption / communication
    {'watch', 'inspect', 'read', 'check', 'look', 'point', 'teach',
     'greet', 'wave', 'eat', 'eat_at', 'drink', 'sip', 'smell',
     'toast', 'blow', 'buy', 'pay', 'serve', 'dribble', 'smile'},
    # 5: no interaction / other (fallback)
    {'no_interaction', 'none', 'no_contact', 'not_contact', 'other'},
]


def build_verb_hierarchy(
    verb_names: List[str], dataset: str = 'hicodet',
) -> Tuple[torch.Tensor, int]:
    """Map fine-grained verb indices to coarse group indices.

    Uses keyword matching against ``_COARSE_GROUPS``.  Verbs that do
    not match any group are assigned to the last (fallback) group.

    Returns:
        fine_to_coarse: ``LongTensor`` of shape ``(num_fine,)``
        num_coarse: total number of coarse groups
    """
    num_groups = len(_COARSE_GROUPS)
    fine_to_coarse = torch.zeros(len(verb_names), dtype=torch.long)

    for idx, verb in enumerate(verb_names):
        v = verb.lower().strip().replace(' ', '_')
        assigned = False
        for gid, keywords in enumerate(_COARSE_GROUPS):
            if v in keywords:
                fine_to_coarse[idx] = gid
                assigned = True
                break
        if not assigned:
            for gid, keywords in enumerate(_COARSE_GROUPS):
                for kw in keywords:
                    if kw in v or v in kw:
                        fine_to_coarse[idx] = gid
                        assigned = True
                        break
                if assigned:
                    break
        if not assigned:
            fine_to_coarse[idx] = num_groups - 1

    return fine_to_coarse, num_groups
