from dataclasses import dataclass
from abc import ABC

@dataclass
class BaseConfig(ABC):
    """
    Configuration class for the GPT model.
    """
