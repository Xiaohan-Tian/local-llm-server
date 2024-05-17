import os
import platform

def detect_os():
    os_type = platform.system().lower()
    if 'linux' in os_type:
        return 'linux'
    elif 'darwin' in os_type:
        return 'darwin'
    elif 'windows' in os_type:
        return 'windows'
    else:
        return 'unknown'

def determine_gpu(use_gpu=True):
    if not use_gpu:
        return 'cpu'
    
    os_type = platform.system().lower()
    if 'linux' in os_type:
        return 'cuda'
    elif 'darwin' in os_type:
        return 'mps'
    elif 'windows' in os_type:
        return 'cuda'
    else:
        return 'cpu'
    
def multi_line_input(prompt=""):
    """Custom function to capture multi-line input."""
    print(prompt, end="", flush=True)
    lines = []
    while True:
        try:
            line = input()
            if line == "":
                break
            lines.append(line)
        except EOFError:
            break
    return "\n".join(lines).strip()

def multi_line_input_with_stop_words(prompt="", stop_words=None):
    """Custom function to capture multi-line input with stop word handling."""
    if stop_words is None:
        stop_words = []
    print(prompt, end="", flush=True)
    lines = []
    while True:
        try:
            line = input()
            # Check if the line starts with any stop word
            if any(line.startswith(word) for word in stop_words):
                lines.append(line)
                break
            lines.append(line)
        except EOFError:
            break
    return "\n".join(lines).strip()

def split_content_and_command(user_input):
    # Check if there is at least one newline in the input
    if '\n' in user_input:
        # Find the last occurrence of a newline
        last_newline_index = user_input.rfind('\n')
        # Extract everything before the last newline
        user_content = user_input[:last_newline_index]
        # Extract the last line after the last newline
        user_command = user_input[last_newline_index + 1:]
    else:
        # If there is no newline, the content is None and the command is the whole input
        user_content = None
        user_command = user_input
        
    if user_content is None: user_content = ""
    if user_command is None: user_command = ""

    return user_content.strip(), user_command.strip()

def convert_path(my_path):
    # Expand the user's home directory symbol (~)
    expanded_path = os.path.expanduser(my_path)

    # Normalize the path for the current operating system
    normalized_path = os.path.normpath(expanded_path)

    # Optional: Convert to absolute path
    absolute_path = os.path.abspath(normalized_path)

    return absolute_path

def load_file_content(file_path):
    """
    Load the content of a file if the file_path is not empty or null and the file exists.
    
    :param file_path: The path to the file to be loaded
    :return: The content of the file, or a message indicating the file does not exist
    """
    if file_path:
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                content = file.read()
            return content
        else:
            return ""
    else:
        return ""

