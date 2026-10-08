# Declared Input Files

Engine-declared file options connect configuration-relative paths to worker input
paths and content-sensitive step fingerprints. Engines expose the
`InputFileOptions` and optional `InputFileDependencies` capabilities from the
[engine contract](engines_api.md); this module resolves their declarations
without teaching the configuration or cache how to parse engine-specific formats.

Resolution uses the YAML file's source directory and retains the spelling the
user wrote. It does not open files, so validation and dry-run can resolve inputs
that an earlier step will produce. Preparation checks that direct files and any
declared manifest payloads are readable. Cache identity includes their contents;
changing an NPZ payload referenced by a quantum descriptor invalidates the
consuming step even when the descriptor path is unchanged.

See [Declared files and artifact validation](../developer/adding-an-engine.md#declared-files-and-artifact-validation)
for authoring the capabilities, and [Quantum data handoffs](../engines/qiskit-handoffs.md)
for their use by both Qiskit engines.

::: chemrefine.input_files
    options:
      members: [OptionPath, option_pointer, resolve_input_file_options, normalized_input_options, declared_input_files, validate_input_files]
