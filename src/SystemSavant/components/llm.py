from SystemSavant.logging import logger
from SystemSavant.entity import LLMConfig
from SystemSavant.constants import QUERY_PROMPT, SYSTEM_PROMPT, PERSIST_DIR, HF_TOKEN
from torch import float16
import os
from huggingface_hub import login

login(token=HF_TOKEN)

from pathlib import Path
from SystemSavant.utils.common import load_key
from SystemSavant.logging import logger
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, StorageContext
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import chromadb

class LLM:
    
    def __init__(self, config: LLMConfig):
        
        self.config = config
        if not self.config.hf_token:
            logger.info("HF Token not found in config. Trying to Load it again.")
            self.config.hf_token = load_key(HF_TOKEN)
        
        self.query_engine = None
        
    def prepare_llm(self):
        
        logger.info("Initialising Chromadb")
        db = chromadb.PersistentClient(path=PERSIST_DIR)
        chroma_client = chromadb.EphemeralClient()
        chroma_collection = chroma_client.create_collection("SSV")
        logger.info("Chromadb successfully initialised")
        
        logger.info("Initiating Document Reader")
        query_wrap_prompt = SimpleInputPrompt(QUERY_PROMPT)
        documents = SimpleDirectoryReader(
            Path(self.config.data_path)
        ).load_data() 
        logger.info("Successfully loaded data from Documents")
        
        logger.info("Initiating Storage Context using Chroma")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection, db=db)
        storage_context = self.load_storage_context(vector_store)
        logger.info("Successfully loaded storage context")
        
        logger.info("Initialising Service Context")
        service_context = self.load_service_context(query_wrap_prompt)
        logger.info("Successfully loaded Service Context")
        
        logger.info("Creating Vector Indexes")
        index = VectorStoreIndex.from_documents(
            documents, storage_context= storage_context, service_context= service_context,
        )
        logger.info("Successfully created a Index")
        
        logger.info("Trying to Start a query engine out of created Index")
        
        query_engine = index.as_query_engine()
        
        if self.query_engine:
            logger.info("Query engine is absent")
            logger.info("Reloading Query")
            
        
        self.query_engine = query_engine
        
        
    def load_storage_context(self, vector_store):
        return StorageContext.from_defaults(vector_store=vector_store)
    
    def load_service_context(self, query_wrapper_prompt):
        llm = HuggingFaceLLM(
            context_window= self.config.context_window,
            max_new_tokens= self.config.max_new_tokens,
            model_name= self.config.model_name,
            generate_kwargs={"temperature": self.config.temperature, "do_sample": True},
            system_prompt= SYSTEM_PROMPT,
            query_wrapper_prompt=query_wrapper_prompt,
            tokenizer_name= self.config.model_name,
            device_map= self.config.device,
            model_kwargs= {"torch_dtype":float16,
                           
                           "token": self.config.hf_token,
                           }
        )
        
        embedd_model = LangchainEmbedding(
            HuggingFaceEmbeddings(
                model_name= self.config.embedding,
            )
        )
        
        service_context = ServiceContext.from_defaults(
            llm = llm,
            embed_model = embedd_model,
            chunk_size = self.config.chunk_size,
        )
        
        return service_context
        
    def generate_response(self, query: str) -> str:
        
        if not self.query_engine:
            logger.error("Query Engine not found. Please initialise it first")
            self.prepare_llm()
            
        response = self.query_engine.query(query)
        return response
        