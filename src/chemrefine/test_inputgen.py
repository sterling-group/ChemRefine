"""
test_inputgen.py

Minimal sanity test for inputgen.py + Ollama backend.
"""

import yaml

from inputgen import (
    OllamaClient,
    generate_with_autocorrect,
    ollama_health_check,
)

def main():
    # 1) Check Ollama is reachable
    print("Checking Ollama server...")
    ollama_health_check()
    print("✓ Ollama is running\n")

    # 2) Create LLM client
    llm = OllamaClient(
        model="llama3.1:8b",   # or qwen2.5:7b-instruct
        temperature=0.0,
    )

    # 3) Example user request (free text)
    user_request = """
    Start from step1.xyz.
    Step 1: GOAT with DFT, keep 25 structures.
    Step 2: OPT+SP with MLFF using Boltzmann sampling.
    Step 3: OPT+SP with DFT refinement.
    Step 4: Train an MLFF model on GPU with 10 percent validation.
    """

    print("User request:")
    print(user_request)
    print("=" * 60)

    # 4) Generate workflow config (dict)
    cfg = generate_with_autocorrect(
    user_request=user_request,
    llm=llm,
    require_paths=False,
    max_attempts=10,
    verbose=True,
)


    # 5) Dump YAML to screen
    yaml_text = yaml.safe_dump(cfg, sort_keys=False)

    print("Generated workflow YAML:")
    print("=" * 60)
    print(yaml_text)

    # 6) Optional: write to file
    with open("input.generated.yaml", "w") as f:
        f.write(yaml_text)

    print("✓ Saved to input.generated.yaml")


if __name__ == "__main__":
    main()




