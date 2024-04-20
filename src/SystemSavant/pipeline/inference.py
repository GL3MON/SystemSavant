from SystemSavant.logging import logger
from SystemSavant.config.configuration import ConfigurationManager
from SystemSavant.components.llm import LLM
from time import time

class Chat:
    
    def __init__(self):    
        self.config_manager = ConfigurationManager()
        
    def initiate(self):
        
        logger.info("Initialising LLM")
        config = self.config_manager.get_llm_config()
        self.llm = LLM(config)
        self.llm.prepare_llm()
        logger.info("LLM Successfully Initialised")
        
    def respond(self, query:str):
        
        start = time()
        response = self.llm.generate_response(query)
        end = time()
        
        logger.info(f"Response generated in {end-start} seconds")
        
        return response