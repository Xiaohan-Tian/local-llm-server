import requests
import time
from flask import Blueprint, jsonify, request

from util.ConfigLoader import ConfigLoader
from llm.LLM import LLM

route_completions = Blueprint('route_completions', __name__)

@route_completions.route('/chat/completions', methods=['POST'])
def completions():
    config = ConfigLoader().get()
    model_config = config['model_config']
    default_completion_config = model_config['default_completion_config']
    debug_mode = config['debug_mode']
    
    default_max_tokens = default_completion_config['max_tokens']
    default_temperature = default_completion_config['temperature']
    default_repeat_penalty = default_completion_config['repeat_penalty']
    default_echo = default_completion_config['echo']
    default_top_p = default_completion_config['top_p']
    
    data = request.get_json()
    messages = data.get('messages', [])
    max_tokens = data.get('max_tokens', default_max_tokens)
    temperature = data.get('temperature', default_temperature)
    top_p = data.get('top_p', default_top_p)
    
    if debug_mode:
        print(f"=== ===  === ===  === ===\t\tCOMPLETE\t\t=== ===  === ===  === ===")
        print(f"max_tokens \t\t = {max_tokens}")
        print(f"temperature \t\t = {temperature}")
        print(f"top_p \t\t\t = {top_p}")
        print(f"default_repeat_penalty \t = {default_repeat_penalty}")
        print(f"default_echo \t\t = {default_echo}")
    
    if not messages:
        return jsonify({'error': 'No messages provided'}), 400
    
    llm = LLM.get()
    
    text = llm.complete_messages(
        messages, 
        max_tokens=max_tokens,
        temperature=temperature,
        repeat_penalty=default_repeat_penalty,
        echo=default_echo
    )
    
    # Format the response in the same way as OpenAI's API
    response = {
        'id': '0',
        'object': 'text_completion',
        'created': int(time.time()),
        'model': f"{model_config['hf_id']} - {model_config['hf_file']}",
        'choices': [{
            # 'text': text,
            "message": {
                "content": text,
                "role": "assistant"
            },
            'index': 0, 
            'logprobs': None, 
            'finish_reason': 'length'
        }]
    }
    return jsonify(response)
