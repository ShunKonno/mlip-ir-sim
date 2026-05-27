from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class SystemConfig:
    xyz_path: Path
    num_molecules: int
    model: str | None = None
    mode: Literal["bulk"] = "bulk"


class SystemBuilder:
    """
    将来的に、単一分子XYZからPBCセルを構築し、ASE Atoms を返すビルダー。
    現時点ではREADMEのAPI形状を固定するための骨格のみ。
    """

    def __init__(self, xyz_path: str | Path, num_molecules: int, model: str | None = None):
        self.config = SystemConfig(
            xyz_path=Path(xyz_path),
            num_molecules=int(num_molecules),
            model=model,
        )

    def build(self):
        raise NotImplementedError("SystemBuilder.build() is not implemented yet.")

