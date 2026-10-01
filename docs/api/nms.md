# Normal-Mode Sampling

The engine-independent two-round NMS coordinator — displacement maths,
`NmsOptions`, the run/rebuild/reattempt orchestration, and the reuse fingerprint.
It drives any engine through the single [`NmsCapableEngine`](engines_api.md)
``nms_input_info`` hook — quoting its ``frequency_request`` when an input asks for no
frequencies — and reads the frequency values (`frequencies` /
`imaginary_freqs` / `normal_modes`) off each parsed structure — never re-parsing an output.

::: chemrefine.nms
