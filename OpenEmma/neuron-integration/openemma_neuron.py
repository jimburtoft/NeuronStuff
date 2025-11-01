"""
OpenEMMA with AWS Neuron/Inferentia2 Support

This module provides Neuron-optimized model loading and inference for OpenEMMA.
Use model path containing 'neuron' to trigger this code path.

Example usage:
    python main.py --model-path meta-llama/Llama-3.2-11B-Vision-Instruct-neuron
"""

import torch
from PIL import Image

# Neuron-specific imports
try:
    from neuronx_distributed_inference.models.mllama.modeling_mllama import (
        NeuronMllamaForCausalLM,
        MllamaInferenceConfig
    )
    from neuronx_distributed_inference.models.config import (
        MultimodalVisionNeuronConfig,
        OnDeviceSamplingConfig
    )
    from neuronx_distributed_inference.utils.hf_adapter import (
        HuggingFaceGenerationAdapter,
        load_pretrained_config
    )
    from neuronx_distributed_inference.models.mllama.utils import (
        create_vision_mask,
        get_image_tensors,
        add_instruct
    )
    from transformers import AutoProcessor, GenerationConfig
    NEURON_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Neuron libraries not available: {e}")
    print("Please activate the Neuron virtual environment:")
    print("  source /opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/bin/activate")
    NEURON_AVAILABLE = False


def load_neuron_model(model_path, compiled_model_path=None):
    """
    Load a Neuron-compiled Llama model for Inferentia2.
    
    Args:
        model_path: HuggingFace model path (e.g., "meta-llama/Llama-3.2-11B-Vision-Instruct")
        compiled_model_path: Path to compiled model (default: /home/ubuntu/compiled_models/llama-3.2-11b-vision-tp8)
    
    Returns:
        tuple: (model, processor, tokenizer=None)
    """
    if not NEURON_AVAILABLE:
        raise RuntimeError("Neuron libraries not available. Cannot load Neuron model.")
    
    # Remove '-neuron' suffix if present
    base_model_path = model_path.replace('-neuron', '').replace('_neuron', '')
    
    # Default compiled model path
    if compiled_model_path is None:
        compiled_model_path = "/home/ubuntu/compiled_models/llama-3.2-11b-vision-tp8"
    
    print(f"Loading Neuron model from: {compiled_model_path}")
    
    # Configure on-device sampling
    on_device_sampling_config = OnDeviceSamplingConfig(dynamic=True)
    
    # Configure Neuron settings
    neuron_config = MultimodalVisionNeuronConfig(
        tp_degree=8,
        batch_size=1,
        max_context_length=1024,
        seq_len=2048,
        on_device_sampling_config=on_device_sampling_config,
        enable_bucketing=True,
        sequence_parallel_enabled=True,
        fused_qkv=True,
        async_mode=False,
    )
    
    config = MllamaInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(base_model_path)
    )
    
    # Load the compiled model - CRITICAL: use load() method
    model = NeuronMllamaForCausalLM(compiled_model_path)
    model.load(compiled_model_path)
    
    # Load processor from base model
    processor = AutoProcessor.from_pretrained(base_model_path)
    
    print(f"✓ Neuron model loaded successfully")
    return model, processor, None


def neuron_vlm_inference(text, images, model, processor, max_new_tokens=2048):
    """
    Run inference using Neuron-optimized Llama model.
    
    Args:
        text: Prompt text
        images: Path to image file
        model: Neuron model instance
        processor: HuggingFace processor
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        str: Generated text response
    """
    if not NEURON_AVAILABLE:
        raise RuntimeError("Neuron libraries not available.")
    
    # Load and prepare image
    image = Image.open(images).convert('RGB')
    batch_size = 1
    num_img_per_prompt = 1
    batch_image = [[image] * num_img_per_prompt] * batch_size
    
    # Get image tensors using NxD helper
    pixel_values, aspect_ratios, num_chunks, has_image = get_image_tensors(
        model.config, batch_image
    )
    
    # Prepare prompt using NxD helper
    formatted_prompt = add_instruct(text, has_image)
    batch_prompt = [formatted_prompt] * batch_size
    
    # Tokenize
    inputs = processor.tokenizer(
        batch_prompt, 
        padding=True, 
        return_tensors="pt", 
        add_special_tokens=False
    )
    
    # Create vision mask
    vision_token_id = processor.tokenizer("<|image|>", add_special_tokens=False).input_ids[0]
    vision_mask = create_vision_mask(inputs.input_ids, vision_token_id)
    
    # Prepare generation config
    generation_config = GenerationConfig.from_pretrained(
        model.config.load_config._name_or_path
    )
    generation_config.update(top_k=1)
    
    # Use HuggingFace generation adapter
    from neuronx_distributed_inference.modules.generation.sampling import prepare_sampling_params
    
    generation_model = HuggingFaceGenerationAdapter(model)
    sampling_params = prepare_sampling_params(
        batch_size=batch_size,
        top_k=[1],
        top_p=[1.0],
        temperature=[1.0]
    )
    
    # Generate
    outputs = generation_model.generate(
        inputs.input_ids,
        generation_config=generation_config,
        attention_mask=inputs.attention_mask,
        max_length=model.config.neuron_config.max_length,
        sampling_params=sampling_params,
        pixel_values=pixel_values,
        aspect_ratios=aspect_ratios,
        vision_mask=vision_mask,
        num_chunks=num_chunks,
        has_image=has_image,
        max_new_tokens=max_new_tokens,
    )
    
    # Decode output
    output_text = processor.tokenizer.batch_decode(
        outputs, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )[0]
    
    # Extract assistant response
    import re
    matches = re.findall(
        r'<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>',
        output_text,
        re.DOTALL
    )
    if matches:
        return matches[0].strip()
    
    return output_text


# Export functions
__all__ = ['load_neuron_model', 'neuron_vlm_inference', 'NEURON_AVAILABLE']
