# Local LLM Server

## Overview
This application is a local server that emulates the OpenAI completion API, allowing you to use your own large language models (LLMs) for inference. It supports both CUDA and CPU-based inference, making it versatile for different hardware setups. The server is built using Flask and is compatible with OpenAI's API request and response format. Additionally, this server is compatible with WSGI servers (e.g., Gunicorn) in a production environment. See the `Usage` section for details.

## Features
- Emulates OpenAI's completion API locally.
- Supports CUDA and CPU inference.
- Compatible with custom LLM models.
- Uses Flask for easy web server setup.

## Installation

### Prerequisites
- Python 3.9.16
- Pip package manager

### Steps
1. **Install `llama-cpp-python`:**
    
    a. Linux
    ```bash
    CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.2.74 --force-reinstall --upgrade --no-cache-dir --verbose
    ```
    b. macOS
    - Make sure you have XCode installed (Please install the full version via AppStore, not only the command-line tools)
    - Install `conda` via miniforge3
   ```bash
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh
    ```
    - for Intel CPU w/AMD GPU based macOS:
    ```bash
    pip uninstall llama-cpp-python -y
    CMAKE_ARGS="-DLLAMA_METAL=on" pip install -U llama-cpp-python==0.2.74 --no-cache-dir
    ```
    - for Apple Silicon based macOS:
    ```bash
    pip uninstall llama-cpp-python==0.2.74 -y
    CMAKE_ARGS="-DLLAMA_METAL_EMBED_LIBRARY=ON -DLLAMA_METAL=on" pip install -U llama-cpp-python --no-cache-dir
    ```
    c. Windows
    - Install [**Visual Studio**](https://visualstudio.microsoft.com/vs/community/).
    - Install [**CUDA Toolkits**](https://developer.nvidia.com/cuda-downloads), please make sure the CUDA Toolkits version you are trying to install can match the CUDA version display in the `nvidia-smi` command's result, or just let the CUDA Toolkits installer to override your existing NVIDIA driver.
    - Install `llama-cpp-python`:
    ```
    set CMAKE_ARGS=-DLLAMA_CUBLAS=on
    set FORCE_CMAKE=1

    echo %CMAKE_ARGS%
    echo %FORCE_CMAKE%

    pip install llama-cpp-python==0.2.74 --force-reinstall --upgrade --no-cache-dir --verbose
    ```

   For detailed installation instructions, please refer to the [**llama-cpp-python**](https://github.com/abetlen/llama-cpp-python) project page.
2. **Install Other Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Starting the Server
1. **Using Gunicorn (Recommended for production):**
    ```bash
    ENV=prod MODEL=mistral gunicorn --worker-class gthread --threads 4 --bind 0.0.0.0:8000 'app:start_server()'
    ```

2. **Using Flask (For local testing):**
    ```bash
    python app.py --model mistral --port 8000
    ```

### Making Requests
The server uses the same request and response format as the OpenAI completion API. Here's an example using `curl`:
1. **Linux**
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -H "User-Agent: insomnia/8.6.1" \
  -d '{
        "messages": [
            {
                "role": "user",
                "content": "When is the day with the longest daylight of the year?"
            }
        ]
      }' \
  "http://localhost:8000/v1/chat/completions"
```

2. **Windows** (Require Windows 10/11 or install [**cURL for Windows**](https://curl.se/windows/))
```
curl -X POST -H "Content-Type: application/json" -H "User-Agent: insomnia/8.6.1" -d "{\"messages\": [{\"role\": \"user\", \"content\": \"When is the day with the longest daylight of the year?\"}]}" "http://localhost:8000/v1/chat/completions"
```

## Configuration
The server can be configured using environment variables and a YAML configuration file. Refer to the `config` directory for example configurations.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

