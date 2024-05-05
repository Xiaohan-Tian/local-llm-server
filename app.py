import os
import sys
import json
import yaml
import argparse
import threading
import uvicorn
from fastapi import FastAPI

from util.ConfigLoader import ConfigLoader
from util.Loggers import print_centered
from loader.HFLoader import load_model
from communicator.LLMCommunicator import LLMCommunicator
from chatbot.Chatbot import Chatbot

from routes.route_hi import router as route_hi_router
from routes.route_completions import router as route_completions_router

main_lock = threading.Lock()


# wrapper logic for WSGI
# example: 
# pip install gunicorn
# ENV=test MODEL=mistral gunicorn --worker-class gthread --threads 4 --bind 127.0.0.1:8000 'app:start_server()'
# NOTE: 
# - avoid using Multi-Process worker in order to prevent multiple LLM instances.
# - in an environment doesn't allow binding external external-accessible ports (e.g. Paperspace), bind "127.0.0.1" instead of "0.0.0.0"
def start_server(model=os.getenv('MODEL')):
    global main_lock
    
    if model is None:
        raise ValueError("model can't be empty")

    config = ConfigLoader().load_config(model).get()

    formatted_json = json.dumps(config, indent=4)
    print(f"=== ===  === ===  === ===\t\t CONFIGURATIONS \t\t=== ===  === ===  === ===")
    print(formatted_json)

    # load model
    load_model()

    # initiate LLM
    print(f"=== ===  === ===  === ===\t\t INIT LLM \t\t=== ===  === ===  === ===")
    LLMCommunicator.get()

    # start the server
    print(f"=== ===  === ===  === ===\t\t SERVER STARTED \t\t=== ===  === ===  === ===")

    app = FastAPI()
    app.include_router(route_hi_router, prefix=config['url_prefix'])
    app.include_router(route_completions_router, prefix=config['url_prefix'])
    
    if main_lock.locked():
        main_lock.release()

    return app

def start_uvicorn():
    if args.chatbot == 1 and config['show_log'] == 0:
        print("starting server...")
        log_config = uvicorn.config.LOGGING_CONFIG
        log_config['handlers']['default']['level'] = 'CRITICAL'
        config['debug_mode'] = False
        
    app = start_server(model=args.model)
    uvicorn.run(app, host=config['host'], port=int(config['port']))
        
# local test server
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process model parameter.')
    parser.add_argument('--model', required=True, help='Model name to load configuration for')
    parser.add_argument('--host', required=False, help='The host which the server should use')
    parser.add_argument('--port', required=False, type=int, help='The port which the server should listen to')
    parser.add_argument('--use_gpu', required=False, type=int, help='Should use GPU')
    
    parser.add_argument('--chatbot', required=False, type=int, default=0, help='Enable CLI Chatbot')
    parser.add_argument('--multiline', required=False, type=int, default=0, help='Allow multi-line input for commandline chatbot')
    parser.add_argument('--show_log', required=False, type=int, default=0, help='show system logs in the Chatbot mode')

    args = parser.parse_args()    

    config = ConfigLoader().load_config(args.model).get()

    if args.host is not None:
        config['host'] = args.host
        
    if args.port is not None:
        config['port'] = args.port

    if args.use_gpu is not None:
        config['use_gpu'] = (True if args.use_gpu == 1 else False)
        
    if args.multiline is not None:
        config['multiline'] = (True if args.multiline == 1 else False)
        
    if args.show_log is not None:
        config['show_log'] = (True if args.show_log == 1 else False)
    
    if args.chatbot == 0:
        start_uvicorn()
    else:
        main_lock.acquire()
        
        server_thread = threading.Thread(target=start_uvicorn)
        server_thread.start()
    
    if args.chatbot == 1:
        main_lock.acquire()
        chatbot = Chatbot()
        chatbot.run()
    