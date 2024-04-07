import threading

from communicator.LLMCommunicator import LLMCommunicator
from communicator.MistralCommunicator import MistralCommunicator

from util.ConfigLoader import ConfigLoader


class LLM:
    _instance = None
    _lock = threading.Lock()
    
    @staticmethod
    def get():
        with LLM._lock:
            if LLM._instance is None:
                # initial the LLMCommunicator
                config = ConfigLoader().get()
                model_config = config['model_config']
                debug_mode = config['debug_mode']
                
                hf_id = model_config['hf_id']
                hf_file = model_config['hf_file']
                n_threads = model_config['n_threads']
                n_batch = model_config['n_batch']
                n_gpu_layers = (model_config['n_gpu_layers'] if config['use_gpu'] else 0)
                n_ctx = model_config['n_ctx']
                system_prompt = model_config['system_prompt']
                verbose = model_config['verbose']
                
                target_model_path = f"{config['model_root']}/{config['model_config']['hf_id']}/{config['model_config']['hf_file']}"
                
                if debug_mode:
                    print(f"hf_id \t\t\t = {hf_id}")
                    print(f"hf_file \t\t = {hf_file}")
                    print(f"n_threads \t\t = {n_threads}")
                    print(f"n_batch \t\t = {n_batch}")
                    print(f"n_gpu_layers \t\t = {n_gpu_layers}")
                    print(f"n_ctx \t\t\t = {n_ctx}")
                    print(f"system_prompt \t\t = {system_prompt}")
                    print(f"verbose \t\t = {verbose}")
                    print(f"target_model_path \t = {target_model_path}")
                    
                LLM._instance = MistralCommunicator(
                    model_path=target_model_path,
                    n_threads=n_threads,
                    n_batch=n_batch,
                    n_gpu_layers=n_gpu_layers,
                    n_ctx=n_ctx,
                    verbose=verbose
                )
                
                LLM._instance.load_model()

            return LLM._instance
