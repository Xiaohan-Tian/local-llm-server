from abc import ABC, abstractmethod
from llama_cpp import Llama

from communicator.LLMCommunicator import LLMCommunicator

class MistralCommunicator(LLMCommunicator):
    def __init__(self, model_path='', n_threads=2, n_batch=512, n_gpu_layers=-1, n_ctx=8196, verbose=True):
        super().__init__(model_path, n_threads, n_batch, n_gpu_layers, n_ctx, verbose)
        
        
    def load_model(self):
        self._llm = Llama(
            model_path=self._model_path,
            n_threads=self._n_threads, # CPU cores
            n_batch=self._n_batch, # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
            n_gpu_layers=self._n_gpu_layers, # Change this value based on your model and your GPU VRAM pool.
            n_ctx=self._n_ctx, # Context window
        )
    
    
    def complete(self, prompt, max_tokens=8196, temperature=0.0, repeat_penalty=1.1, stop='', echo=True):
        if self._verbose: 
            print(f"prompt\t\t= {prompt}")
            print(f"stop\t\t= '{stop}'")
        
        response = self._llm(
            prompt=prompt,
            max_tokens=self._n_ctx,
            temperature=temperature,
            repeat_penalty=repeat_penalty,
            stop = [] if stop == "" else [stop], # Dynamic stopping when such token is detected.
            echo=True # return the prompt
        )
        
        response_text = response.get("choices", [{}])[0].get("text", "")
        response_text = response_text.strip()  # Trim the response_text

        if response_text.startswith(prompt):
            response_text = response_text[len(prompt):].strip()  # Remove prompt and strip any leading/trailing whitespace
        
        if self._verbose: 
            print(f"response\t= {response_text}")
            
        return response_text
    
    
    def get_prompt(self, messages):
        prompt = ''
        stop_token = ''
        
        for i in range(len(messages)):
            current_message = messages[i]
            current_role = current_message.get('role', 'user')
            current_content = current_message.get('content', '')
            
            if current_role == 'user':
                current_role = 'user'
            elif current_role == 'assistant':
                current_role = 'assistant'
            else:
                continue
            
            if i <= 1:
                stop_token = '</s>'
                if current_role == 'user':
                    prompt += f"<s>[INST] {current_content} [/INST]"
                else:
                    prompt += f" {current_content} </s>"
            else:
                stop_token = ''
                if current_role == 'user':
                    prompt += f"[INST] {current_content} [/INST]"
                else:
                    prompt += f" {current_content}"
                    
        return prompt, stop_token
