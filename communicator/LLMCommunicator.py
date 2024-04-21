import threading

from llama_cpp import Llama

from util.ConfigLoader import ConfigLoader

class LLMCommunicator:
    _lock = threading.Lock()
    _instance = None
    
    def __init__(self):
        self.config = ConfigLoader().get()
        
        config = ConfigLoader().get()
        model_config = config['model_config']
        debug_mode = config['debug_mode']
        
        hf_id = model_config['hf_id']
        hf_file = model_config['hf_file']

        self._config = config
        self._debug_mode = debug_mode
        self._model_path = f"{config['model_root']}/{config['model_config']['hf_id']}/{config['model_config']['hf_file']}"
        self._n_threads = model_config['n_threads']
        self._n_batch = model_config['n_batch']
        self._n_gpu_layers = (model_config['n_gpu_layers'] if config['use_gpu'] else 0)
        self._n_ctx = model_config['n_ctx']
        self._verbose = model_config['verbose']
        
        self.system_prompt = model_config['system_prompt']
        self.system_prompt_start_token = model_config['system_prompt_start_token']
        self.system_prompt_end_token = model_config['system_prompt_end_token']
        self.user_prompt_start_token = model_config['user_prompt_start_token']
        self.user_prompt_end_token = model_config['user_prompt_end_token']
        self.user_followup_prompt_start_token = model_config['user_followup_prompt_start_token']
        self.user_followup_prompt_end_token = model_config['user_followup_prompt_end_token']
        self.assistant_prompt_start_token = model_config['assistant_prompt_start_token']
        self.assistant_prompt_end_token = model_config['assistant_prompt_end_token']
        self.assistant_followup_prompt_start_token = model_config['assistant_followup_prompt_start_token']
        self.assistant_followup_prompt_end_token = model_config['assistant_followup_prompt_end_token']
        self.end_tokens = model_config['end_tokens']
        
        if self._debug_mode:
            print(f"hf_id \t\t\t = {hf_id}")
            print(f"hf_file \t\t = {hf_file}")
            print(f"n_threads \t\t = {self._n_threads}")
            print(f"n_batch \t\t = {self._n_batch}")
            print(f"n_gpu_layers \t\t = {self._n_gpu_layers}")
            print(f"n_ctx \t\t\t = {self._n_ctx}")
            print(f"verbose \t\t = {self._verbose}")
            print(f"target_model_path \t = {self._model_path}")
            print(f"system_prompt                             = {self.system_prompt}")
            print(f"system_prompt_start_token                 = {self.system_prompt_start_token}")
            print(f"system_prompt_end_token                   = {self.system_prompt_end_token}")
            print(f"user_prompt_start_token                   = {self.user_prompt_start_token}")
            print(f"user_prompt_end_token                     = {self.user_prompt_end_token}")
            print(f"user_followup_prompt_start_token          = {self.user_followup_prompt_start_token}")
            print(f"user_followup_prompt_end_token            = {self.user_followup_prompt_end_token}")
            print(f"assistant_prompt_start_token              = {self.assistant_prompt_start_token}")
            print(f"assistant_prompt_end_token                = {self.assistant_prompt_end_token}")
            print(f"assistant_followup_prompt_start_token     = {self.assistant_followup_prompt_start_token}")
            print(f"assistant_followup_prompt_end_token       = {self.assistant_followup_prompt_end_token}")
            print(f"end_tokens                                = {self.end_tokens}")
        
        self._llm = None

    def load_model(self):
        self._llm = Llama(
            model_path=self._model_path,
            n_threads=self._n_threads,
            n_batch=self._n_batch,
            n_gpu_layers=self._n_gpu_layers,
            n_ctx=self._n_ctx,
        )

    def _full_complete(self, prompt, max_tokens=8196, temperature=0.0, repeat_penalty=1.1, echo=True):
        stop = self.end_tokens
        
        if self._verbose: 
            print(f"prompt\t\t= {prompt}")
            print(f"stop\t\t= '{stop}'")
        
        response = self._llm(
            prompt=prompt,
            max_tokens=self._n_ctx,
            temperature=temperature,
            repeat_penalty=repeat_penalty,
            stop=stop,
            echo=echo
        )
        
        response_text = response.get("choices", [{}])[0].get("text", "").strip()
        
        if self._verbose: 
            print(f"response\t= {response_text}")
        
        return response_text

    def _stream_complete(self, prompt, max_tokens=8196, temperature=0.0, repeat_penalty=1.1, echo=True):
        stop = self.end_tokens
        
        if self._verbose: 
            print(f"streaming mode")
            print(f"prompt\t\t= {prompt}")
            print(f"stop\t\t= '{stop}'")
        
        response_stream = self._llm.create_completion(
            prompt=prompt,
            max_tokens=self._n_ctx,
            temperature=temperature,
            repeat_penalty=repeat_penalty,
            stop=stop,
            echo=echo,
            stream=True
        )
        
        return response_stream

    def complete_messages(self, messages, max_tokens=8196, temperature=0.0, repeat_penalty=1.1, echo=True, stream=False):
        with LLMCommunicator._lock:
            prompt = self.get_prompt(messages)

            if stream:
                response = self._stream_complete(prompt, max_tokens=max_tokens, temperature=temperature, repeat_penalty=repeat_penalty, echo=echo)
                return response
            else:
                response = self._full_complete(prompt, max_tokens=max_tokens, temperature=temperature, repeat_penalty=repeat_penalty, echo=echo)
                return response
    
    def get_prompt(self, messages):
        prompt = ''
        
        if len(messages) < 1:
            raise "messages list is empty"
            
        current_message = messages[0]
        current_role = current_message.get('role', 'user')
        current_content = current_message.get('content', '').strip()
        
        if self.system_prompt and current_role != "system":
            messages.insert(0, {
                "role": "system", 
                "content": ""
            })
            
        # print(messages)
        
        is_first_user_prompt = True
        is_first_assistant_prompt = True
        
        for i in range(len(messages)):
            current_message = messages[i]
            # print(current_message)
            current_role = current_message.get('role', 'user')
            current_content = current_message.get('content', '').strip()
            
            if current_role == 'system':
                if i == 0:
                    prompt += f"{self.system_prompt_start_token}{current_content}{self.system_prompt_end_token}"
                else:
                    raise "system prompt can only be set at the beginning"
                    
                # print(f"(seen: system) prompt = {prompt}")
            elif current_role == 'user':
                if is_first_user_prompt:
                    prompt += f"{self.user_prompt_start_token}{current_content}{self.user_prompt_end_token}"
                    is_first_user_prompt = False
                else:
                    prompt += f"{self.user_followup_prompt_start_token}{current_content}{self.user_followup_prompt_end_token}"
                    
                # print(f"(seen: user) prompt = {prompt}")
            elif current_role == 'assistant':
                if is_first_assistant_prompt:
                    prompt += f"{self.assistant_prompt_start_token}{current_content}{self.assistant_prompt_end_token}"
                    is_first_assistant_prompt = False
                else:
                    prompt += f"{self.assistant_followup_prompt_start_token}{current_content}{self.assistant_followup_prompt_end_token}"
                    
                # print(f"(seen: assistant) prompt = {prompt}")
            else:
                raise f"unknow role: {current_role}"
                    
        return prompt

    def is_system_prompt_supported(self):
        return self.system_prompt
    
    @staticmethod
    def get():
        with LLMCommunicator._lock:
            if LLMCommunicator._instance is None:
                LLMCommunicator._instance = LLMCommunicator()
                LLMCommunicator._instance.load_model()
                
            return LLMCommunicator._instance
