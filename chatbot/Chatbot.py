import os
import re
import sys
import json
import threading
import traceback
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live

from util.ConfigLoader import ConfigLoader
from util.Loggers import print_centered, fill_row
from util.Utilities import multi_line_input_with_stop_words, split_content_and_command
from communicator.LLMCommunicator import LLMCommunicator
from loader.HFLoader import load_model

class Chatbot:
    _lock = threading.Lock()
    
    def __init__(self):
        with Chatbot._lock:
            os.environ['TOKENIZERS_PARALLELISM'] = 'False'

            """Initialize the chatbot with the necessary API key."""
            self.config = ConfigLoader().get()
            self.console = Console()
            self.messages = []
            self.i18n = ConfigLoader().load_config(self.config['language'], path='./i18n_config/').get()

            # always print as soon as possible
            formatted_json = json.dumps(self.config, indent=4)
            if self.config['debug_mode']:
                print(f"CONFIGURATIONS: ")
                print(formatted_json)
            
    def complete(self, messages=[]):
        with Chatbot._lock:
            config = ConfigLoader().get()
            model_config = config['model_config']
            default_completion_config = model_config['default_completion_config']
            llm = LLMCommunicator.get()
            stream_batch_size = config['stream_batch_size']
            context = ""
            
            default_max_tokens = default_completion_config['max_tokens']
            default_temperature = default_completion_config['temperature']
            default_repeat_penalty = default_completion_config['repeat_penalty']
            default_echo = default_completion_config['echo']
            
            # print(f"default_max_tokens = {default_max_tokens}")
            # print(f"default_temperature = {default_temperature}")
            # print(f"default_repeat_penalty = {default_repeat_penalty}")
            # print(f"default_echo = {default_echo}")

            # Check if messages are provided and length is appropriate
            res = llm.complete_messages(
                messages, 
                max_tokens=default_max_tokens,
                temperature=default_temperature,
                repeat_penalty=default_repeat_penalty,
                echo=default_echo,
                stream=True
            )

            # stream mode
            response_stream = res
            def generate(batch_size=stream_batch_size):
                prev_item = next(response_stream, None)  # Get the first item
                bulk_text = ""
                current_batch_size = 0
                first_item = True

                for item in response_stream:
                    # option 1: directly yeild the current value
                    # yield get_response_json(prev_item.choices[0].delta.content)

                    # option 2: yield values in bulk
                    bulk_text += prev_item['choices'][0]['text']
                    current_batch_size += 1

                    if current_batch_size == batch_size:
                        if first_item:
                            yield bulk_text
                            first_item = False
                        else:
                            yield bulk_text

                        bulk_text = ""
                        current_batch_size = 0

                    prev_item = item

                # Handle the last item
                if prev_item is not None:
                    # option 1: directly yeild the current value
                    # print(f"token[END] = {item['choices'][0]['delta']['content']}")

                    # option 2: yield values in bulk
                    bulk_text += prev_item['choices'][0]['text']
                    yield bulk_text

            return generate()

    def add_message(self, role, content):
        """
        Add a message to the conversation history.

        If the message role is 'user' and the content starts with 'system:', 
        change the role to 'system' and adjust the content to what follows 
        after 'system:', trimming any spaces.

        Parameters:
            role (str): The role of the message ('user' or 'system').
            content (str): The content of the message.

        """
        
        # Check if the role is 'user' and the content starts with 'system:'
        proceed_response = True
        if role == 'user' and content.startswith('system:'):
            role = 'system'  # Change the role to 'system'
            content = content[7:].strip()  # Update content to exclude 'system:' and trim spaces
            proceed_response = False

        # Append the message to the conversation history
        self.messages.append({"role": role, "content": content})
        return proceed_response

    def stream_openai_response(self):
        """Generate responses from OpenAI and stream them to the console."""
        try:
            completion = self.complete(self.messages)

            # Collect chunks to process markdown partially
            fill_row('─')
            self.console.print(f"{self.i18n['cb_assistant']}:\n", style="bold #AB68FF")

            partial_response = ""
            with Live(Markdown(""), console=self.console, auto_refresh=True) as live:
                for chunk in completion:
                    text = chunk
                    if text:
                        partial_response += text
                        # Update live display with partial markdown
                        live.update(Markdown(partial_response))

            # Append full AI response to history
            self.add_message("assistant", partial_response)

        except Exception as e:
            self.console.print(f"Error: {e}", style="bold red")
            
    def switch_language(self, language):
        print(f"Please restart the app to reload the language resource file.")
        
        # update config
        new_default_config = ConfigLoader.read_config('default.yaml')
        new_default_config['language'] = language
        # print(new_default_config)
        ConfigLoader.save_config('default.yaml', config=new_default_config)
        
    def load_model(self, model_name):
        print(f"Loading model {model_name}...")
        
        # Offload
        LLMCommunicator.pop()
        
        # update config
        new_default_config = ConfigLoader.read_config('default.yaml')
        new_default_config['model'] = model_name
        # print(new_default_config)
        ConfigLoader.save_config('default.yaml', config=new_default_config)
        
        # reload config
        config = ConfigLoader(reload=True).get()
        config = ConfigLoader().load_config(model_name, path='./llm_config/').get()
        
        formatted_json = json.dumps(config, indent=4)
        print(f"=== ===  === ===  === ===\t\t CONFIGURATIONS \t\t=== ===  === ===  === ===")
        print(formatted_json)
        
        # initiate LLM
        LLMCommunicator.get()

    def pull_model(self, path1, path2, path3, config_name, base_config_name):
        print(f"Pulling model from {path1}/{path2}/{path3} as {config_name} from {base_config_name}...")
        
        # create config
        new_llm_config = ConfigLoader.read_config(f"{base_config_name}.yaml", path="./llm_config/")
        new_llm_config['model_config']['hf_id'] = f"{path1}/{path2}"
        new_llm_config['model_config']['hf_file'] = f"{path3}"
        ConfigLoader.save_config(f"{config_name}.yaml", path="./llm_config/", config=new_llm_config)
        
        # pull model from HF
        load_model(model_config=new_llm_config['model_config'])

    def run(self):
        """Run the chatbot, accepting input until the user types 'quit'."""
        if self.config['multiline']:
            self.console.print(self.i18n['cb_help'], style="bold green")
        else:
            self.console.print(self.i18n['cb_help'], style="bold green")
            
        while True:
            fill_row('─')
            self.console.print(f"{self.i18n['cb_you']}: ", style="bold #d2bc95")

            user_input = ""
            if self.config['multiline']:
                user_input = multi_line_input_with_stop_words("\n\n>>> ", ['/quit', '/q', '/clear', '/c', '/load', '/pull', '/chat', '/']) 
            else:
                user_input = input("\n\n>>> ") 

            user_content, user_command = split_content_and_command(user_input)
            
            if user_command == "/quit" or user_command == "/q":
                os._exit(0)
                break
            elif user_command == "/clear" or user_command == "/c":
                self.messages = []
                self.console.print(self.i18n['cb_chat_history_cleared'], style="bold green")
            elif match := re.match(r"/language ([a-zA-Z][a-zA-Z0-9_-]*)$", user_command):
                language = match.group(1)
                self.switch_language(language)
            elif match := re.match(r"/load ([a-zA-Z][a-zA-Z0-9_-]*)$", user_command):
                model_name = match.group(1)
                self.load_model(model_name)
            elif match := re.match(r"/pull ([a-zA-Z][a-zA-Z0-9_.-]*)/([a-zA-Z][a-zA-Z0-9_.-]*)/([a-zA-Z][a-zA-Z0-9_.-]*) as ([a-zA-Z][a-zA-Z0-9_-]*) from ([a-zA-Z][a-zA-Z0-9_-]*)$", user_command):
                path1, path2, path3, config_name, base_config_name = match.groups()
                self.pull_model(path1, path2, path3, config_name, base_config_name)
            elif user_command in ['/chat', '/']:
                proceed_response = self.add_message("user", user_content)
                if proceed_response:
                    try:
                        self.stream_openai_response()
                    except Exception as e:
                        print("An error occurred while running the chatbot:")
                        self.console.print(e)
            else:
                self.console.print(self.i18n['cb_unknown_command'], style="bold red")
