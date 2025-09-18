from .s3tokenizer import (
    S3_SR,
    S3_HOP,
    S3_TOKEN_HOP,
    S3_TOKEN_RATE,
    SPEECH_VOCAB_SIZE,
    S3Tokenizer,
)


SOS = SPEECH_VOCAB_SIZE
EOS = SPEECH_VOCAB_SIZE + 1



def drop_invalid_tokens(x):
    """Drop SoS and EoS - supports batch processing"""
    # Handle different input shapes
    if len(x.shape) == 1:
        x = x.unsqueeze(0)  # Add batch dimension
    elif len(x.shape) > 2:
        raise ValueError(f"Expected 1D or 2D tensor, got {len(x.shape)}D")

    batch_size = x.shape[0]
    batch_results = []

    for i in range(batch_size):
        seq = x[i]

        # Find start position (after SOS token)
        if SOS in seq:
            s = (seq == SOS).nonzero(as_tuple=True)[0]
            if len(s) > 0:
                s = s[0].item() + 1
            else:
                s = 0
        else:
            s = 0

        # Find end position (before EOS token)
        if EOS in seq:
            e = (seq == EOS).nonzero(as_tuple=True)[0]
            if len(e) > 0:
                e = e[0].item()
            else:
                e = None
        else:
            e = None

        # Extract valid tokens
        processed_seq = seq[s:e]
        batch_results.append(processed_seq)

    # Return single result for batch_size=1 (backward compatibility)
    return batch_results[0] if batch_size == 1 else batch_results
