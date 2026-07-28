"""Language prompt templates for autonomously-collected episodes.

pi0.5 conditions on the prompt string, so logging the same literal sentence
for every episode would teach it one string rather than the concept.
plan_shuffle() already tells us the color and whether the destination is a
calibrated marker or open table -- enough to vary phrasing and target
description without any extra vision work.
"""

from __future__ import annotations

import random

_TO_MARKER = [
    "put the {color} cube on the marker",
    "place the {color} block on the tag",
    "move the {color} cube onto the marker",
    "set the {color} block down on the marker",
]

_TO_SLOT = [
    "put the {color} cube on the table",
    "move the {color} block to an open spot",
    "place the {color} cube on the work surface",
    "clear the {color} block onto the table",
]


def build_prompt(color: str, place_kind: str | None) -> str:
    templates = _TO_MARKER if place_kind == "to_marker" else _TO_SLOT
    return random.choice(templates).format(color=color)


# Stacking distinguishes the first cube (bare marker -- placed like any
# table-to-marker move) from every level after (a cube goes on TOP OF
# another cube, a materially different visual target), so the two get
# separate template pools rather than one that's vague about which.
_STACK_BASE = [
    "place the {color} cube on the marker to start a stack",
    "put the {color} block down on the tag",
    "set the {color} cube on the marker as the base",
]

_STACK_ON_TOP = [
    "stack the {color} block on top",
    "add the {color} cube to the stack",
    "put the {color} block on top of the tower",
    "place the {color} cube on top of the stack",
]

_UNSTACK = [
    "take the top block off the stack",
    "unstack the {color} cube",
    "remove the {color} block from the tower",
    "take the {color} cube off the top and set it down",
]


def build_stack_prompt(color: str, level: int) -> str:
    """Prompt for placing ``color`` as the (1-based) ``level``-th cube."""
    templates = _STACK_BASE if level <= 1 else _STACK_ON_TOP
    return random.choice(templates).format(color=color)


def build_unstack_prompt(color: str) -> str:
    """Prompt for removing the top cube (``color``) from a stack."""
    return random.choice(_UNSTACK).format(color=color)
