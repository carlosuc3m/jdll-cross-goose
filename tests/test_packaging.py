import re
import tomllib
from pathlib import Path

from packaging.requirements import Requirement


def test_upstream_dependency_is_pinned_to_the_headless_only_fork():
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    dependencies = {requirement.name: requirement for requirement in map(Requirement, project["project"]["dependencies"])}
    assert re.fullmatch(r"git\+https://github.com/carlosuc3m/cross-goose\.git@[0-9a-f]{40}",
                        dependencies["crossgoose"].url)
    assert "opencv-python" not in dependencies
    assert "2.0.2" in dependencies["numpy"].specifier
    assert "1.26.4" not in dependencies["numpy"].specifier
