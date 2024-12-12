import importlib.util
import os
import onnx
from torch import onnx as torch_onnx

def load_class_from_file(class_name: str, folder_path: str, file_name: str = None, *args, **kwargs):
    """Dynamically load a class from a Python file."""
    # Determine the file path
    if file_name and os.path.isabs(file_name):
        # Use absolute path if file_name is already an absolute path
        file_path = file_name
    elif file_name and os.path.dirname(file_name):
        # Use relative path directly if file_name contains a directory
        file_path = file_name
    else:
        # Otherwise, construct the path from folder_path and class_name
        file_name = file_name or f"{class_name}.py"
        file_path = os.path.join(folder_path, file_name)

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[ERROR] File '{file_name}' not found in folder '{folder_path}'. Resolved path: {file_path}")

    # Dynamically load the class from the file
    spec = importlib.util.spec_from_file_location(class_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Retrieve and instantiate the class
    if hasattr(module, class_name):
        return getattr(module, class_name)(*args, **kwargs)
    else:
        raise AttributeError(f"[ERROR] Class '{class_name}' not found in file '{file_path}'.")



def parse_dynamic_args(dynamic_args):
    """Parse key-value arguments from argparse."""
    parsed_args = {}
    if dynamic_args:
        for arg in dynamic_args:
            key, value = arg.split('=', 1)
            try:
                parsed_args[key] = eval(value)
            except Exception:
                parsed_args[key] = value
    return parsed_args


def save_onnx(model, test_loader, output_path="quantized_model.onnx"):
    """Save a PyTorch model in ONNX format."""
    # Set model to evaluation

    dummy_input = next(iter(test_loader))[0]
    torch_onnx.export(model, dummy_input, output_path)
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model saved successfully.")

