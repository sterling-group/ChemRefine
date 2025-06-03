import pytest
from chemrefine.core import ChemRefiner

def test_chemrefiner_initialization(monkeypatch, tmp_path):
    """
    Test ChemRefiner initialization and YAML parsing.
    """
    # Create dummy YAML
    yaml_content = """
    charge: 0
    multiplicity: 1
    steps:
      - step: 1
        calculation_type: DFT
        sample_type:
          method: random
          parameters:
            fraction: 0.1
    """
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(yaml_content)

    # Mock ArgumentParser.parse() to return dummy args
    class DummyArgs:
        maxcores = 4
        skip = False
        input_file = str(yaml_file)
    dummy_args = DummyArgs()

    # Patch the ArgumentParser
    from chemrefine import parse
    monkeypatch.setattr(parse.ArgumentParser, "parse", lambda self: (dummy_args, None))

    # Initialize ChemRefiner
    chemrefiner = ChemRefiner()

    assert chemrefiner.charge == 0
    assert chemrefiner.multiplicity == 1
    assert isinstance(chemrefiner.config['steps'], list)
