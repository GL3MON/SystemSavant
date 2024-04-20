from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LLMConfig:
    data_path: Path
    hf_token: str
    model_name: str
    device: str
    embedding: str
    chunk_size : int
    max_new_tokens: int
    context_window: int
    bit_4_quant: bool
    temperature: float