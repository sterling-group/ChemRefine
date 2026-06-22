# Normal-Mode Sampling

The engine-independent two-round NMS coordinator — displacement maths,
`NmsOptions`, the run/rebuild/reattempt orchestration, and the reuse fingerprint.
It drives any engine through the two hooks of
[`NmsCapableEngine`](engines_api.md); a compute engine supplies only those.

::: chemrefine.nms
