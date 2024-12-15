import torch
import logging
from models import create_model
import time
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def benchmark_throughput(model: torch.nn.Module, input_size: Tuple[int, int, int, int], num_iterations: int = 100) -> float:
    """Benchmark model throughput in images/second."""
    device = next(model.parameters()).device
    x = torch.randn(input_size).to(device)
    
    # Warmup
    for _ in range(10):
        model(x)
    
    # Benchmark
    start_time = time.time()
    
    for _ in range(num_iterations):
        model(x)
        
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    images_per_second = (input_size[0] * num_iterations) / elapsed_time
    
    return images_per_second

def test_model(model_name: str, batch_size: int = 2):
    """Test UNetAdapter with a specific backbone"""
    logger.info(f"Testing {model_name}")
    
    # Create model
    model = create_model(model_name, None, pretrained=True)
    input_size = (batch_size, 1, 224, 224)
    
    try:
        x = torch.randn(input_size)
        y = model(x)
        logger.info(f"Output shape: {y.shape}")
        
        # Benchmark throughput
        throughput = benchmark_throughput(model, input_size)
        logger.info(f"Throughput: {throughput:.2f} images/sec\n")
        
        return True
    except Exception as e:
        logger.error(f"Test failed for {model_name}")
        logger.error(f"Error: {str(e)}\n")
        return False

def main():
    # Models to test
    models = [
        'timm/resnet50.a1_in1k',
        'timm/vit_base_patch16_clip_224.openai',
    ]
    
    # Test each model
    results = {}
    for model_name in models:
        results[model_name] = test_model(model_name)
    
    # Print summary
    logger.info("Test Summary:")
    for model_name, passed in results.items():
        status = "✓" if passed else "✗"
        logger.info(f"{status} {model_name}")

if __name__ == "__main__":
    main() 