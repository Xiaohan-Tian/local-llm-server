import os
import sys
import json
import threading
import traceback
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live

from util.ConfigLoader import ConfigLoader
from util.Loggers import print_centered, fill_row
from util.Utilities import multi_line_input
from communicator.LLMCommunicator import LLMCommunicator

class Chatbot:
    _lock = threading.Lock()
    
    def __init__(self):
        with Chatbot._lock:
            os.environ['TOKENIZERS_PARALLELISM'] = 'False'

            """Initialize the chatbot with the necessary API key."""
            self.config = ConfigLoader().get()
            self.console = Console()
            self.messages = []

            # always print as soon as possible
            self.config['stream_batch_size'] = 1
            formatted_json = json.dumps(self.config, indent=4)
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
            self.console.print(f"ASSISTANT:\n", style="bold #AB68FF")

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

    def run(self):
        """Run the chatbot, accepting input until the user types 'quit'."""
        if self.config['multiline']:
            self.console.print("Press 'Enter/Return' to start a new line, press 'Enter/Return' twice to submit the message. Type 'quit' to exit, type 'clear' to clear chat history.", style="bold green")
        else:
            self.console.print("Type 'quit' to exit, type 'clear' to clear chat history.", style="bold green")
            
        while True:
            fill_row('─')
            self.console.print("YOU: ", style="bold #d2bc95")

            user_input = ""
            if self.config['multiline']:
                user_input = multi_line_input("\n\n>>> ") 
            else:
                user_input = input("\n\n>>> ") 

            if user_input.lower() == 'quit':
                os._exit(0)
                break
            elif user_input.lower() == 'clear':
                self.messages = []
                self.console.print("Chat history has been cleared.", style="bold green")
            else:
                proceed_response = self.add_message("user", user_input)
                if proceed_response:
                    try:
                        self.stream_openai_response()
                    except Exception as e:
                        print("An error occurred while running the chatbot:")
                        self.console.print(e)
