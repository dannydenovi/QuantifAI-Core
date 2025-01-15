import os
import torch
import argparse
from torch.utils.data import  DataLoader
from genericDataset import GenericDataset
from quantUtils import evaluate_metrics, quantize_model_fx, quantize_model_dynamic
from utils import load_class_from_file, parse_dynamic_args, save_onnx

# Set device
torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Main Script
if __name__ == "__main__":

    quantization_types = ["static", "dynamic"]
    available_data_types = ["int8", "float32"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--training_set_path", default="training_dataset.pth", help="Path to training dataset")
    parser.add_argument("--test_set_path", default="test_dataset.pth", help="Path to test dataset")
    parser.add_argument("--model_file", required=True, help="Path to model file")
    parser.add_argument("--model_class", required=True, help="Class name of the model")
    parser.add_argument("--model_path", default="trained_model.pth", help="Path to trained model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--input_format", type=str, help="Input format of the model")
    parser.add_argument("--num_batches", type=int, default=1, help="Batches for quantization calibration")
    parser.add_argument("--is_classification", action="store_true", help="If the task is classification")
    parser.add_argument("--save_onnx", action="store_true", help="Save model in ONNX format")
    parser.add_argument("--args", nargs=argparse.REMAINDER)
    parser.add_argument("--test", action="store_true", help="Test saved model")
    parser.add_argument("--quantization_method", default="static", choices=quantization_types, help="Quantization method")
    parser.add_argument("--dtype", default="int8", choices=available_data_types, help="Quantization datatype")
    args = parser.parse_args()

    dynamic_args = parse_dynamic_args(args.args)
    input_format = eval(args.input_format) if args.input_format else None
    is_classification = args.is_classification if args.is_classification else False

    # Load datasets
    train_ds = GenericDataset(args.training_set_path, input_format)
    test_ds = GenericDataset(args.test_set_path, input_format)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Load model
    model = load_class_from_file(args.model_class, os.path.dirname(args.model_file), args.model_file, **dynamic_args)
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))

    # Evaluate raw model
    print("Raw Model Metrics:", evaluate_metrics(test_loader, model, is_classification=is_classification))

    #Data type
    if args.dtype == "int8":
        type = torch.qint8
    else:
        type = torch.float32

    # Quantize model
    if args.quantization_method == "dynamic":
        quantized_model = quantize_model_dynamic(model, train_loader, args.num_batches, type=type)
    else:
        quantized_model = quantize_model_fx(model, train_loader, args.num_batches, type=type)

    # if test dont save
    if not args.test:
        torch.save(quantized_model.state_dict(), "quantized_model.pth")
        print("Quantized model saved successfully.")
    quantized_model.load_state_dict(torch.load("quantized_model.pth", map_location=torch.device('cpu')))

    # Evaluate quantized model
    print("Quantized Model Metrics:", evaluate_metrics(test_loader, quantized_model, is_classification=is_classification))

    # Print models sizes
    raw_size = os.path.getsize(args.model_path)
    quantized_size = os.path.getsize("quantized_model.pth")

    print(f"Raw Model Size: {raw_size / 1024**2 :.2f} Mb")
    print(f"Quantized Model Size: {quantized_size / 1024**2 :.2f} Mb")

    # Save ONNX
    if args.save_onnx:
        save_onnx(quantized_model, test_loader, "quantized_model.onnx")
