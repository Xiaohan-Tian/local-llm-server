import os
import json
import yaml
import argparse
from flask import Flask

from util.ConfigLoader import ConfigLoader
from loader.HFLoader import load_model
from communicator.LLMCommunicator import LLMCommunicator

from routes.route_hi import route_hi
from routes.route_completions import route_completions


# wrapper logic for WSGI
# example: 
# pip install gunicorn
# ENV=test MODEL=mistral gunicorn --worker-class gthread --threads 4 --bind 127.0.0.1:8000 'app:start_server()'
# NOTE: 
# - avoid using Multi-Process worker in order to prevent multiple LLM instances.
# - in an environment doesn't allow binding external external-accessible ports (e.g. Paperspace), bind "127.0.0.1" instead of "0.0.0.0"
def start_server(model=os.getenv('MODEL')):
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
    
    app = Flask(__name__)
    app.register_blueprint(route_hi, url_prefix=config['url_prefix'])
    app.register_blueprint(route_completions, url_prefix=config['url_prefix'])
    
    return app


# local test server
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process model parameter.')
    parser.add_argument('--model', required=True, help='Model name to load configuration for')
    parser.add_argument('--port', required=False, type=int, help='The port which the server should listen to')
    parser.add_argument('--use_gpu', required=False, type=int, help='Should use GPU')

    args = parser.parse_args()    

    config = ConfigLoader().load_config(args.model).get()

    if args.port is not None:
        config['port'] = args.port

    if args.use_gpu is not None:
        config['use_gpu'] = (True if args.use_gpu == 1 else False)
    
    app = start_server(model=args.model)
    app.run(debug=config['debug_mode'], use_reloader=False, host=config['host'], port=config['port'])
    