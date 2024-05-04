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

