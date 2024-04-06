from abc import ABC, abstractmethod
from llama_cpp import Llama


class LLMCommunicator(ABC):
    def __init__(self, model_path, n_threads, n_batch, n_gpu_layers, n_ctx, verbose=True):
        self._model_path = model_path
        self._n_threads = n_threads
        self._n_batch = n_batch
        self._n_gpu_layers = n_gpu_layers
        self._n_ctx = n_ctx
        self._verbose = verbose
        
        self._llm = None

    
    @abstractmethod
    def load_model(self):
        pass
    
    
    @abstractmethod
    def complete(self, prompt, max_tokens=8196, temperature=0.0, repeat_penalty=1.1, stop='', echo=True):
        pass
    
    
    def complete_messages(self, messages, max_tokens=8196, temperature=0.0, repeat_penalty=1.1, echo=True):
        prompt, stop_token = self.get_prompt(messages)
        response = self.complete(prompt, max_tokens=max_tokens, temperature=temperature, repeat_penalty=repeat_penalty, stop=stop_token)
        return response
    
    
    @abstractmethod
    def get_prompt(self, messages):
        pass

    
    def is_system_prompt_supported(self):
        return False
