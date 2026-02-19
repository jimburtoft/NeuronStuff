#!/usr/bin/env python3
"""
Test accuracy consistency between two SigLIP model variants.

This script compares feature vectors from two compiled models to verify that
performance optimizations (like --auto-cast=matmult) don't degrade model accuracy.

Usage:
    python3 test_accuracy.py --model1 MODEL_WITHOUT_FLAGS --model2 MODEL_WITH_FLAGS
    
Example:
    # Compare default vs optimized compilation
    python3 test_accuracy.py \\
        --model1 siglip_384_neuron_default.pt \\
        --model2 siglip_384_neuron.pt

The script generates synthetic test images and compares the cosine similarity
of feature vectors extracted by both models.
"""

import argparse
import json
import numpy as np
import torch
import torch_neuronx
from PIL import Image
import os
import sys


def generate_test_images(num_samples=50, image_size=384, seed=42):
    """Generate reproducible synthetic test images."""
    print(f"Generating {num_samples} test images (seed={seed})...")
    images = []
    np.random.seed(seed)

    for i in range(num_samples):
        # Create varied synthetic images
        img_array = np.random.rand(image_size, image_size, 3)

        # Add some structure (gradients, patterns)
        x_grad = np.linspace(0, 1, image_size)
        y_grad = np.linspace(0, 1, image_size)
        X, Y = np.meshgrid(x_grad, y_grad)

        # Mix random noise with gradients
        pattern = (X + Y) / 2
        img_array = img_array * 0.7 + pattern[:, :, np.newaxis] * 0.3

        # Scale to 0-255
        img_array = (img_array * 255).astype(np.uint8)
        image = Image.fromarray(img_array)
        images.append(image)

    return images


def preprocess_image(image, image_size=384):
    """Preprocess image for SigLIP model."""
    import torchvision.transforms as transforms

    preprocess = transforms.Compose(
        [
            transforms.Resize(
                image_size, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    return preprocess(image).unsqueeze(0)


def test_accuracy_consistency(model1_path, model2_path, num_samples=50):
    """
    Compare feature consistency between two models.

    Args:
        model1_path: Path to first model (baseline)
        model2_path: Path to second model (to compare)
        num_samples: Number of test images to generate

    Returns:
        dict: Consistency metrics
    """
    print("=" * 80)
    print("SigLIP-384 Accuracy Consistency Test")
    print("=" * 80)

    print(f"\nModel 1 (Baseline): {model1_path}")
    print(f"Model 2 (Comparison): {model2_path}")
    print(f"Test samples: {num_samples}")

    # Check if models exist
    if not os.path.exists(model1_path):
        print(f"\n❌ Error: Model file not found: {model1_path}")
        return None

    if not os.path.exists(model2_path):
        print(f"\n❌ Error: Model file not found: {model2_path}")
        return None

    # Load models
    print("\nLoading models...")
    try:
        model1 = torch.jit.load(model1_path)
        model1.eval()
        print(f"  ✓ Loaded model 1")

        model2 = torch.jit.load(model2_path)
        model2.eval()
        print(f"  ✓ Loaded model 2")
    except Exception as e:
        print(f"\n❌ Error loading models: {e}")
        return None

    # Generate test images
    test_images = generate_test_images(num_samples)

    # Extract features from both models
    print(f"\nExtracting features from {num_samples} images...")
    features1 = []
    features2 = []

    with torch.no_grad():
        for i, image in enumerate(test_images):
            try:
                input_tensor = preprocess_image(image)

                # Get features from both models
                feat1 = model1(input_tensor)
                feat2 = model2(input_tensor)

                features1.append(feat1.numpy())
                features2.append(feat2.numpy())

                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{num_samples} images")

            except Exception as e:
                print(f"  ⚠ Warning: Error processing image {i}: {e}")
                continue

    if len(features1) < num_samples * 0.8:  # Require at least 80% success
        print(f"\n❌ Error: Too many failures ({len(features1)}/{num_samples})")
        return None

    # Convert to arrays
    features1 = np.vstack(features1)
    features2 = np.vstack(features2)

    print(f"\nFeature shapes: {features1.shape}")

    # Calculate consistency metrics
    print("\nCalculating consistency metrics...")

    # 1. Mean Absolute Error
    mae = np.mean(np.abs(features1 - features2))

    # 2. Max Absolute Error
    max_error = np.max(np.abs(features1 - features2))

    # 3. Root Mean Square Error
    rmse = np.sqrt(np.mean((features1 - features2) ** 2))

    # 4. Relative Error
    relative_error = np.mean(np.abs(features1 - features2) / (np.abs(features1) + 1e-8))

    # 5. Cosine Similarity (per sample)
    cosine_sims = []
    for f1, f2 in zip(features1, features2):
        norm1 = np.linalg.norm(f1)
        norm2 = np.linalg.norm(f2)
        if norm1 > 0 and norm2 > 0:
            sim = np.dot(f1, f2) / (norm1 * norm2)
            cosine_sims.append(sim)

    avg_cosine_sim = np.mean(cosine_sims)
    min_cosine_sim = np.min(cosine_sims)

    # 6. Check if features are identical (within tolerance)
    tolerance = 1e-5
    identical = np.allclose(features1, features2, rtol=tolerance, atol=tolerance)

    # Compile results
    results = {
        "model1": model1_path,
        "model2": model2_path,
        "num_samples": len(features1),
        "feature_dimension": features1.shape[1],
        "mean_absolute_error": float(mae),
        "max_absolute_error": float(max_error),
        "root_mean_square_error": float(rmse),
        "relative_error": float(relative_error),
        "avg_cosine_similarity": float(avg_cosine_sim),
        "min_cosine_similarity": float(min_cosine_sim),
        "features_identical": bool(identical),
        "tolerance": tolerance,
        "test_passed": bool(avg_cosine_sim > 0.999),
    }

    # Print results
    print("\n" + "=" * 80)
    print("CONSISTENCY METRICS")
    print("=" * 80)
    print(f"\nError Metrics:")
    print(f"  Mean Absolute Error:     {mae:.8f}")
    print(f"  Max Absolute Error:      {max_error:.8f}")
    print(f"  Root Mean Square Error:  {rmse:.8f}")
    print(f"  Relative Error:          {relative_error:.8f}")

    print(f"\nCosine Similarity:")
    print(f"  Average: {avg_cosine_sim:.8f}")
    print(f"  Minimum: {min_cosine_sim:.8f}")

    print(f"\nIdentity Check:")
    print(f"  Features identical (tol={tolerance}): {identical}")

    print(f"\n" + "=" * 80)
    if results["test_passed"]:
        print("✅ ACCURACY TEST PASSED")
        print("   Models produce equivalent features.")
        print(f"   Cosine similarity: {avg_cosine_sim:.5f} (> 0.999 required)")
        print("   The optimization is safe to use.")
    else:
        print("❌ ACCURACY TEST FAILED")
        print("   Models produce different features!")
        print(f"   Cosine similarity: {avg_cosine_sim:.5f} (< 0.999 threshold)")
        print("   Review the optimization settings.")
    print("=" * 80)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test accuracy consistency between two SigLIP models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare default vs optimized model
  python3 test_accuracy.py --model1 siglip_384_neuron_default.pt --model2 siglip_384_neuron.pt
  
  # Use more samples for thorough testing
  python3 test_accuracy.py --model1 model1.pt --model2 model2.pt --num-samples 100
  
  # Save results to file
  python3 test_accuracy.py --model1 model1.pt --model2 model2.pt --output results.json

Interpretation:
  - Cosine similarity > 0.999: Models are equivalent ✅
  - Cosine similarity 0.99-0.999: Small differences, likely acceptable
  - Cosine similarity < 0.99: Significant differences, review needed
        """,
    )
    parser.add_argument(
        "--model1",
        type=str,
        required=True,
        help="Path to baseline model (e.g., compiled without optimization flags)",
    )
    parser.add_argument(
        "--model2",
        type=str,
        required=True,
        help="Path to comparison model (e.g., compiled with --auto-cast=matmult)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of test samples (default: 50)",
    )
    parser.add_argument("--output", type=str, help="Output JSON file for results")

    args = parser.parse_args()

    # Run accuracy test
    results = test_accuracy_consistency(args.model1, args.model2, args.num_samples)

    if results is None:
        return 1

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    # Return exit code based on test result
    return 0 if results["test_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
