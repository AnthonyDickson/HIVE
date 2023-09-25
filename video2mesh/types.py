from pathlib import Path
from typing import Tuple, Union

"""A 2-dimensional size, e.g. image resolution. Generally follows the convention of height first."""
Size = Tuple[int, int]
File = Union[str, Path]
