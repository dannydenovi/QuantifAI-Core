# QuantfAI - CORE

This repository provides a modular framework for model quantization, enabling users to reduce model size and optimize performance without significantly compromising accuracy. It supports both classification and regression tasks, offers seamless evaluation of raw and quantized models, and includes an optional ONNX export for production deployment.

## Key Features

- **Dynamic Model Loading:** Load and initialize models dynamically from external Python files.  
- **Quantization:** Utilize FX Graph Mode Quantization for PyTorch models.  
- **Metrics Evaluation:** Compute metrics such as accuracy, mean squared error (MSE), and R² for both raw and quantized models.  
- **Model Size Comparison:** Analyze size reductions and storage savings from quantization.  
- **ONNX Export:** Optionally export the quantized model to ONNX format for streamlined deployment.

## Requirements

- Python 3.8+  
- PyTorch  
- ONNX  
- argparse

## Installation

```bash
git clone https://github.com/your-repo/quantization-service.git
cd quantization-service
pip install -r requirements.txt
```

## Usage

Run the main script with the following arguments:

```bash
python main.py \
    --training_set_path training_dataset.pth \
    --test_set_path test_dataset.pth \
    --model_file ./models/YourModel.py \
    --model_class YourModelClassName \
    --model_path trained_model.pth \
    --batch_size 32 \
    --input_format "(3, 32, 32)" \
    --num_batches 10 \
    --is_classification \
    --save_onnx \
    --args "hidden_channels=64" "num_layers=4"
```

### Arguments

| Argument               | Description                                              | Default                 |
|------------------------|----------------------------------------------------------|-------------------------|
| `--training_set_path`  | Path to the training dataset (in `.pth` format)          | `training_dataset.pth`  |
| `--test_set_path`      | Path to the test dataset (in `.pth` format)              | `test_dataset.pth`      |
| `--model_file`         | Path to the Python file containing the model class.      | **Required**            |
| `--model_class`        | Name of the class to instantiate from `model_file`.       | **Required**            |
| `--model_path`         | Path to the trained model `.pth` file.                   | `trained_model.pth`     |
| `--batch_size`         | Batch size for training and evaluation.                  | `32`                    |
| `--input_format`       | Input shape (e.g., `(3, 32, 32)` for CIFAR-10).          | `None`                  |
| `--num_batches`        | Number of batches for quantization calibration.           | `1`                     |
| `--is_classification`  | Flag for classification tasks (omit for regression).      | `False`                 |
| `--save_onnx`          | Save the quantized model in ONNX format.                 | `False`                 |
| `--args`               | Additional model initialization arguments (e.g. `hidden_channels=128`). | `None`        |

## Outputs

- **Quantized Model:** Saved as `quantized_model.pth`  
- **Metrics:** Prints raw and quantized model metrics (accuracy, MSE, R²)  
- **Model Sizes:** Displays size comparisons between raw and quantized models  
- **ONNX Model (Optional):** If `--save_onnx` is used, outputs `quantized_model.onnx`

## Example

For a CIFAR-10 classification model:

```bash
python main.py \
    --training_set_path cifar10_train.pth \
    --test_set_path cifar10_test.pth \
    --model_file ./models/LeNet.py \
    --model_class LENET \
    --model_path lenet_trained.pth \
    --batch_size 64 \
    --input_format "(3, 32, 32)" \
    --num_batches 5 \
    --is_classification \
    --save_onnx \
    --args "num_classes=10"
```

## Customization

### Adding Your Model

1. Place your model definition file in the `models/` directory.
2. Ensure the class name matches the `--model_class` argument you provide.

### Extending Metrics

To add new evaluation metrics, modify the `evaluate_metrics` function within `quantUtils.py`.

### Supporting New Data Formats

To handle new dataset formats, extend the `GenericDataset` class as needed.

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests for new features or bug fixes. For major changes, please open an issue first to discuss your ideas.

For questions or support, contact [me](danny.denovi@unime.it).

---
