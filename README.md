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
   ```bash
   CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.2.42 --force-reinstall --upgrade --no-cache-dir --verbose
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
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -H "User-Agent: insomnia/8.6.1" \
  -d '{
        "messages": [
            {
                "role": "user",
                "content": "When is the day longest daytime among the year?"
            }
        ]
      }' \
  "http://localhost:8000/v1/chat/completions"
```

## Configuration
The server can be configured using environment variables and a YAML configuration file. Refer to the `config` directory for example configurations.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

