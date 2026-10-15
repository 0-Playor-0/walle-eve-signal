"""Charging policy. A `charge` action on a hearth takes one turn and refills to 100%, so there is never a
reason to charge less than full: the only question is whether to charge at all."""
from __future__ import annotations


def should_charge(battery: float, on_hearth: bool, required: bool, opportunistic_below: float) -> bool:
    """Charge when the safety rule sent us here, or when we are standing on a hearth anyway and below
    the opportunistic threshold. Never detour to a hearth just to top up."""
    return on_hearth and battery < 100.0 - 1e-9 and (required or battery < opportunistic_below)
