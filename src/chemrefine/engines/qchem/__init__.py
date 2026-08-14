"""Q-Chem engine package.

The pieces land module by module — the input writer first, the output reader and template
inspector next, the registered engine last. Until the engine module exists this package
registers nothing: auto-discovery imports it and finds no ``@register``, which is a no-op
by design.
"""
