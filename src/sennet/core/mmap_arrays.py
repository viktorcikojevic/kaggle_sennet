from typing import *
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import json


@dataclass
class MmapArray:
    data: np.memmap
    shape: List[int]
    metadata: Dict


def read_mmap_array(
        root_path: Path,
        read_json: Optional[Dict] = None,
        mode: str = "c",
) -> MmapArray:
    root_path = Path(root_path)
    if read_json is None:
        read_json = json.loads((root_path / "metadata.json").read_text())
    shape = read_json["shape"]
    dtype = read_json["dtype"]
    data = np.memmap(
        filename=str(root_path / "data.npy"),
        dtype=getattr(np, dtype),
        mode=mode,
        shape=tuple(shape),
    )
    arr = MmapArray(
        data=data,
        shape=shape,
        metadata=read_json,
    )
    return arr


def create_mmap_array(
        output_dir: Path,
        shape: List[int],
        dtype: type,
) -> MmapArray:
    metadata = {
        "shape": shape,
        "dtype": dtype.__name__
    }
    output_dir.mkdir(exist_ok=True, parents=True)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=4))
    data = np.memmap(
        filename=str(output_dir / "data.npy"),
        dtype=dtype,
        mode="write",
        shape=tuple(shape),
    )
    arr = MmapArray(
        data=data,
        shape=shape,
        metadata=metadata,
    )
    return arr
