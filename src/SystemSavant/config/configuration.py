from SystemSavant.utils.common import read_yaml, create_directories
from SystemSavant.logging import logger
from SystemSavant.entity import (LLMConfig,)
from SystemSavant.constants import CONFIG_PATH, PARAMS_PATH
from pathlib import Path
import os

class ConfigurationManager:
    
    def __init__(self, config_filepath:Path = CONFIG_PATH, params_filepath:Path = PARAMS_PATH):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        
        if (not os.path.exists(Path(self.config.artifacts))):
            create_directories([self.config.artifacts])
        
    def get_llm_config(self) -> LLMConfig:
        
        config = self.config
        params = self.params
        
        llm_config = LLMConfig(
            data_path = config.data,
            hf_token = config.HF_TOKEN,
            temperature = params.llm.temperature,
            model_name = params.llm.model_name,
            max_new_tokens = params.llm.max_new_token,
            context_window = params.llm.context_window,
            bit_4_quant = params.llm.bit_4_quant,
            device= params.llm.device,
            embedding= params.embedding.huggingface,
            chunk_size= params.llm.chunk_size,
        )
        
        return llm_config