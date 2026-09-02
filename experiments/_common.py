"""Shared helpers for the recursive-loop feature-generation modules
(synthetic_features.py, leaf_index_features.py). Factored out because E3
requires both recycling schemes to sit in an IDENTICAL loop -- if seeding or
the mtry rule diverged between them, any difference in collapse behaviour
could be an artifact of that divergence rather than of continuous-vs-discrete
recycling, which is the thing E3 is actually meant to isolate.
"""

import numpy as np


def child_seeds(random_state, n):
    """n independent, deterministic integer seeds derived from one seed, so a
    whole sweep (or a whole generation of the recursive loop) is reproducible
    from a single `random_state`."""
    return [int(child.generate_state(1)[0])
            for child in np.random.SeedSequence(random_state).spawn(n)]


def resolve_mtry(mtry, p):
    """mtry as an int, given a feature count p. mtry="paper" reproduces
    Ishwaran & Malley's own choice: mtry = [p/3], where [z] denotes the first
    integer STRICTLY greater than z -- not ceil(p/3), which differs from that
    when p is divisible by 3 (e.g. p=9: "first integer greater than 3" is 4,
    but ceil(3)=3). floor(z)+1 is the first-integer-strictly-greater-than-z
    identity for any real z, hence p // 3 + 1 here.
    """
    if mtry != "paper":
        return mtry
    return p // 3 + 1
