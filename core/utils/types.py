import numpy as np
from typing import NewType, Optional, NamedTuple, List, Tuple, Dict, Callable, Union
from pathlib import Path

Stream = NewType("Stream", np.ndarray)
Labels = NewType("Label", np.ndarray)
Frame = NewType("Frame", np.ndarray)
