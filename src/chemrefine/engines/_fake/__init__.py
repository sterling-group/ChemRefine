"""Fake engine — importing this module registers it in the ENGINES registry."""

from chemrefine.engines._fake.engine import FakeEngine

__all__ = ["FakeEngine"]
