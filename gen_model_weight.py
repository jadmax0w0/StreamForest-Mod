import sys
sys.path.extend(['.', './llava', './llava/model', './llava/model/mymod', '..'])

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
from typing import Optional


def custom_weight_init(module: nn.Module):
    for name, submodule in module.named_modules():
        if hasattr(submodule, "weight_init"):
            submodule.weight_init()
            print(name, submodule.__class__.__name__, "- initialized")


def check_params_consistency(module: nn.Module, threshold: float = 5.0):
    consistent = True
    inconsis_params = []
    for name, param in module.named_parameters():
        if torch.isinf(param).any() or torch.isnan(param).any() or (torch.abs(param) > threshold).any():
            # print(f"Parameter {name} is probably inconsistent: \n{param}")
            print(f"Parameter {name} is probably inconsistent")
            consistent = False
            inconsis_params.append(name)
    return consistent, inconsis_params


def model_weight_init(model: LlavaQwenForCausalLM, tokenizer: PreTrainedTokenizer, initialize_params: Optional[str] = None, manual: bool = True):
    def check_param_name(name: str, params: list[str]):
        if not params:
            return True
        return any((name in pname or pname in name) for pname in params)
    
    if initialize_params is not None:
        import re
        params = initialize_params.strip()
        params = re.sub(r'\s+', '', initialize_params)
        params = initialize_params.split(',')
    else:
        params = None

    print("Checking parameters consistency")
    consistent, inconsis_params = check_params_consistency(model, 500)

    if not consistent:
        print("Detected incomplete model")
    else:
        cmd = input("You have already loaded the model's complete weight, keep re-initializing? (y/debug/[n])").lower()
        if cmd == "debug":
            import pdb
            pdb.set_trace()
            pass
            return
        elif cmd != "y":
            return
    
    initialize_params = (params + inconsis_params) if params is not None else inconsis_params
    print(f"Paramters to initialize: {initialize_params}")

    if not manual:
        # Setup plug-in module
        print(f"Initializing {model.__class__.__name__}'s weights")
        for name, module in model.named_modules():
            if hasattr(module, "weight_init") and check_param_name(name):
                print(name, module.__class__.__name__)
                module.weight_init()
        print(f"Initialization complete")
    else:
        print("Manual mode to initialize parameters")
        print("You can use `custom_weight_init(nn.Module)` call to initialize a module and all its sub-modules")
        import pdb
        pdb.set_trace()
        pass

    print("Start saving model")
    output_path = input("Output path: ")
    import os, time
    os.makedirs(output_path, exist_ok=True)
    print(f"Model will be saved at {os.path.abspath(output_path)}")
    print("Saving model checkpoint...", end=" ")
    s = time.time()
    model.save_pretrained(output_path)
    t = time.time()
    print(f"Complete, time consumption: {(t - s):.2f}")
    print("Saving tokenizer checkpoint...", end=" ")
    s = time.time()
    tokenizer.save_pretrained(output_path)
    t = time.time()
    print(f"Complete, time consumption: {(t - s):.2f}")

    cmd = input("Exit? y/[n]")
    if cmd.lower() == 'y':
        exit(0)


if __name__ == "__main__":
    raise NotImplementedError("You should not run this code alone")
    import argparse, os, time
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-path", type=str, required=True, help="/path/to/checkpoint/qwen25vl")
    parser.add_argument("-o", "--output-path", type=str, required=True, help="/path/to/output/qwen25vl_o2o")
    parser.add_argument("-p", "--params", type=str, default="")
    parser.add_argument("--manual", action="store_true")
    args = parser.parse_args()

    model_path: str = args.model_path
    output_path: str = args.output_path
    params: str = args.params

    import re
    params = params.strip()
    params = re.sub(r'\s+', '', params)
    params = params.split(',')

    model, processor = load_model(model_path, params, args.manual)
    
    os.makedirs(output_path, exist_ok=True)
    print("Saving model checkpoint...", end=" ")
    s = time.time()
    model.save_pretrained(output_path)
    t = time.time()
    print(f"Complete, time consumption: {(t - s):.2f}")
    print("Saving processor checkpoint...", end=" ")
    s = time.time()
    processor.save_pretrained(output_path)
    t = time.time()
    print(f"Complete, time consumption: {(t - s):.2f}")
    pass