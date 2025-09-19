#!/usr/bin/env python3
"""
Debug script to understand the unpacking error in T3 batch processing.
"""

import torch
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from chatterbox import ChatterboxTTS


def debug_unpacking_error():
    """Debug the unpacking error in T3 batch processing."""
    print("Loading ChatterboxTTS model...")
    tts = ChatterboxTTS.from_pretrained("cpu")

    # Prepare test data
    texts = ["Hello world!", "How are you?"]
    print(f"Input texts: {texts}")

    try:
        # Get the T3 model
        t3_model = tts.t3

        # Mock the parameters as they would be in the failing method
        batch_size = len(texts)

        # These should be the parameters being passed to _inference_parallel_batch
        t3_cond_batch = [tts.conds.t3] * batch_size if tts.conds else []
        text_tokens_batch = [torch.tensor([[1, 2, 3, 4]])] * batch_size
        initial_speech_tokens_batch = None
        temperatures = [0.8] * batch_size
        top_ps = [0.95] * batch_size
        min_ps = [0.05] * batch_size
        length_penalties = [1.0] * batch_size
        repetition_penalties = [1.2] * batch_size
        cfg_weights = [0.5] * batch_size

        print(f"Batch size: {batch_size}")
        print(f"t3_cond_batch length: {len(t3_cond_batch)}")
        print(f"text_tokens_batch length: {len(text_tokens_batch)}")
        print(f"temperatures length: {len(temperatures)}")
        print(f"top_ps length: {len(top_ps)}")
        print(f"min_ps length: {len(min_ps)}")
        print(f"length_penalties length: {len(length_penalties)}")
        print(f"repetition_penalties length: {len(repetition_penalties)}")
        print(f"cfg_weights length: {len(cfg_weights)}")

        # Prepare initial_speech_tokens_batch as done in the method
        if initial_speech_tokens_batch is None:
            initial_speech_tokens_batch = []
            for text_tokens in text_tokens_batch:
                initial_speech = t3_model.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
                initial_speech_tokens_batch.append(initial_speech)

        print(f"initial_speech_tokens_batch length: {len(initial_speech_tokens_batch)}")

        # Test the zip function as done in _inference_parallel_batch (FIXED VERSION)
        args_list = list(zip(
            range(len(t3_cond_batch)),
            t3_cond_batch, text_tokens_batch, initial_speech_tokens_batch,
            temperatures, top_ps, min_ps, length_penalties,
            repetition_penalties, cfg_weights
        ))
        print(f"Args list length: {len(args_list)}")
        if args_list:
            print(f"First args item: {type(args_list[0])}")
            print(f"First args item length: {len(args_list[0])}")
            print(f"First args item content: {args_list[0]}")

        # Test the unpacking that's failing
        if args_list:
            for i, args_item in enumerate(args_list):
                print(f"Item {i}: type={type(args_item)}, length={len(args_item)}")
                try:
                    idx, t3_cond, text_tokens, initial_speech, temp, top_p, min_p, length_pen, rep_pen, cfg_weight = args_item
                    print(f"  Unpacking successful for item {i}")
                except ValueError as e:
                    print(f"  Unpacking failed for item {i}: {e}")
                    print(f"  Args content: {args_item}")

    except Exception as e:
        print(f"Error during debug: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_unpacking_error()