# ./loader/HFLoader.py

import os
import shutil
from huggingface_hub import hf_hub_download

from util.ConfigLoader import ConfigLoader


def load_model(model_config=None):
    config = ConfigLoader().get()
    if model_config is None:
        model_config = config['model_config']
    target_model_folder = f"{config['model_root']}/{model_config['hf_id']}/"
    target_model_path = f"{config['model_root']}/{model_config['hf_id']}/{model_config['hf_file']}"
    
    print(f"=== ===  === ===  === ===\t\tLOADING MODEL\t\t=== ===  === ===  === ===")
    print(f"target_model_path \t = {target_model_path}")
    print(f"hf_id \t\t\t = {model_config['hf_id']}")
    print(f"hf_name \t\t = {model_config['hf_file']}")
    
    if os.path.exists(target_model_path):
        print("File exists, skipping downloading.")
    else:
        print("File does not exist, download from HuggingFace...")
        
        # create the target folder if not exist
        root = config['model_root']
        destination = target_model_folder

        # Remove the root part from the destination path
        relative_destination = os.path.relpath(destination, root)
        print(relative_destination)

        # Split the relative destination path into its components
        subfolders = relative_destination.split(os.sep)
        print(subfolders)

        # Create each subfolder level by level
        current_path = root
        for subfolder in subfolders:
            current_path = os.path.join(current_path, subfolder)
            if not os.path.exists(current_path):
                os.makedirs(current_path)
                print(f"Created folder: {current_path}")

        # download model file and move to the destination
        model_path = hf_hub_download(repo_id=model_config['hf_id'], filename=model_config['hf_file'])

        actual_file = os.path.realpath(model_path)

        print('actual_file =', actual_file)
        shutil.move(actual_file, target_model_path)
