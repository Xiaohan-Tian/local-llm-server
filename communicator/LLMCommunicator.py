import threading

from llama_cpp import Llama

from util.ConfigLoader import ConfigLoader

class LLMCommunicator:
    _lock = threading.Lock()
    
    def __init__(self, model_path='', n_threads=2, n_batch=512, n_gpu_layers=-1, n_ctx=8196, verbose=True):
        self._model_path = model_path
        self._n_threads = n_threads
        self._n_batch = n_batch
        self._n_gpu_layers = n_gpu_layers
        self._n_ctx = n_ctx
        self._verbose = verbose
        
        self._llm = None

    def load_model(self):
        self._llm = Llama(
            model_path=self._model_path,
            n_threads=self._n_threads,
            n_batch=self._n_batch,
            n_gpu_layers=self._n_gpu_layers,
            n_ctx=self._n_ctx,
        )

    def _full_complete(self, prompt, max_tokens=8196, temperature=0.0, repeat_penalty=1.1, stop='', echo=True):
        if self._verbose: 
            print(f"prompt\t\t= {prompt}")
            print(f"stop\t\t= '{stop}'")
        
        response = self._llm(
            prompt=prompt,
            max_tokens=self._n_ctx,
            temperature=temperature,
            repeat_penalty=repeat_penalty,
            stop=[] if stop == "" else [stop],
            echo=echo
        )
        
        response_text = response.get("choices", [{}])[0].get("text", "").strip()
        
        if self._verbose: 
            print(f"response\t= {response_text}")
        
        return response_text

    def _stream_complete(self, prompt, max_tokens=8196, temperature=0.0, repeat_penalty=1.1, stop='', echo=True):
        if self._verbose: 
            print(f"streaming mode")
            print(f"prompt\t\t= {prompt}")
            print(f"stop\t\t= '{stop}'")
        
        response_stream = self._llm.create_completion(
            prompt=prompt,
            max_tokens=self._n_ctx,
            temperature=temperature,
            repeat_penalty=repeat_penalty,
            stop=[] if stop == "" else [stop],
            echo=echo,
            stream=True
        )
        
        return response_stream

    def complete_messages(self, messages, max_tokens=8196, temperature=0.0, repeat_penalty=1.1, echo=True, stream=False):
        with LLMCommunicator._lock:
            prompt, stop_token = self.get_prompt(messages)

            if stream:
                response = self._stream_complete(prompt, max_tokens=max_tokens, temperature=temperature, repeat_penalty=repeat_penalty, stop=stop_token, echo=echo)
                return response
            else:
                response = self._full_complete(prompt, max_tokens=max_tokens, temperature=temperature, repeat_penalty=repeat_penalty, stop=stop_token, echo=echo)
                return response
    
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

    def is_system_prompt_supported(self):
        return False
