import requests
import time
import json
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse

from util.ConfigLoader import ConfigLoader
from communicator.LLMCommunicator import LLMCommunicator

router = APIRouter()


def complete_completions(messages, max_tokens, temperature, repeat_penalty, echo):
    config = ConfigLoader().get()
    model_config = config['model_config']
    default_completion_config = model_config['default_completion_config']
    debug_mode = config['debug_mode']
    
    llm = LLMCommunicator.get()
    
    text = llm.complete_messages(
        messages, 
        max_tokens=max_tokens,
        temperature=temperature,
        repeat_penalty=repeat_penalty,
        echo=echo,
        stream=False
    )
    
    # Format the response in the same way as OpenAI's API
    response = {
        'id': '0',
        'object': 'chat.completion',
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
            'finish_reason': 'stop'
        }]
    }
    return response


def stream_completions(messages, max_tokens, temperature, repeat_penalty, echo):
    config = ConfigLoader().get()
    model_config = config['model_config']
    default_completion_config = model_config['default_completion_config']
    debug_mode = config['debug_mode']
    
    stream_batch_size = config['stream_batch_size']
    
    if debug_mode:
        print(f"stream_batch_size \t = {stream_batch_size}")
    
    llm = LLMCommunicator.get()
    
    response_stream = llm.complete_messages(
        messages, 
        max_tokens=max_tokens,
        temperature=temperature,
        repeat_penalty=repeat_penalty,
        echo=echo,
        stream=True
    )
    
    def get_response_json(chunk_text, is_first=False, is_last=False):
        chunk = {
            'id': '0',
            'object': 'chat.completion.chunk',
            'created': int(time.time()),
            'model': f"{model_config['hf_id']} - {model_config['hf_file']}",
            'choices': [{
                "delta": {
                    "content": chunk_text
                },
                'index': 0, 
                'logprobs': None
            }],
            'finish_reason': None
        }
        
        if is_first:
            chunk['choices'][0]['delta']['role'] = "assistant"
            
        if is_last:
            chunk['finish_reason'] = "stop"
        
        chunk_json = json.dumps(chunk)
        # print(f"!!! chunk_json = {chunk_json}")
        
        return f"data: {chunk_json}\n\n"

    
    def generate(batch_size=stream_batch_size):
        # yield 'Hello '
        # time.sleep(1)
        # yield 'World '
        # time.sleep(1)
        # yield '!'
        
        prev_item = next(response_stream, None)  # Get the first item
        bulk_text = ""
        current_batch_size = 0
        first_item = True
        
        for item in response_stream:
            # option 1: directly yeild the current value
            # yield get_response_json(prev_item['choices'][0]['text'])
            
            # option 2: yield values in bulk
            bulk_text += prev_item['choices'][0]['text']
            current_batch_size += 1
            
            if current_batch_size == batch_size:
                if first_item:
                    yield get_response_json(bulk_text, is_first=True)
                    first_item = False
                else:
                    yield get_response_json(bulk_text)
                    
                bulk_text = ""
                current_batch_size = 0
            
            prev_item = item

        # Handle the last item
        if prev_item is not None:
            # option 1: directly yeild the current value
            # print(f"token[END] = {item['choices'][0]['text']}")
            
            # option 2: yield values in bulk
            bulk_text += prev_item['choices'][0]['text']
            yield get_response_json(bulk_text, is_last=True)
        
        yield f"data: [DONE]\n\n"
        
    return StreamingResponse(generate(), media_type="text/event-stream")
    

@router.post("/chat/completions")
async def completions(request: Request):
    config = ConfigLoader().get()
    model_config = config['model_config']
    default_completion_config = model_config['default_completion_config']
    debug_mode = config['debug_mode']
    
    default_max_tokens = default_completion_config['max_tokens']
    default_temperature = default_completion_config['temperature']
    default_repeat_penalty = default_completion_config['repeat_penalty']
    default_echo = default_completion_config['echo']
    default_top_p = default_completion_config['top_p']
    
    data = await request.json()
    messages = data.get('messages', [])
    max_tokens = data.get('max_tokens', default_max_tokens)
    temperature = data.get('temperature', default_temperature)
    top_p = data.get('top_p', default_top_p)
    stream_mode = data.get('stream', False)
    
    if debug_mode:
        print(f"=== ===  === ===  === ===\t\tCOMPLETE\t\t=== ===  === ===  === ===")
        print(f"max_tokens \t\t = {max_tokens}")
        print(f"temperature \t\t = {temperature}")
        print(f"top_p \t\t\t = {top_p}")
        print(f"default_repeat_penalty \t = {default_repeat_penalty}")
        print(f"default_echo \t\t = {default_echo}")
        print(f"stream_mode \t\t = {stream_mode}")
    
    if not messages:
        return jsonify({'error': 'No messages provided'}), 400
    
    llm = LLMCommunicator.get()
    
    if stream_mode:
        res = stream_completions(
            messages, 
            max_tokens=max_tokens,
            temperature=temperature,
            repeat_penalty=default_repeat_penalty,
            echo=default_echo
        )

        return res
    else:
        res = complete_completions(
            messages, 
            max_tokens=max_tokens,
            temperature=temperature,
            repeat_penalty=default_repeat_penalty,
            echo=default_echo
        )

        return res
