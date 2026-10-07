"""Pre-download every model used by the text pipeline (runs while coding)."""
import time
from huggingface_hub import snapshot_download

MODELS = [
    "openai/whisper-large-v3-turbo",          # ASR with word timestamps (MIT)
    "facebook/wav2vec2-large-960h-lv60-self", # CTC ASR, no LM: keeps learner errors (Apache-2.0)
    "vennify/t5-base-grammar-correction",     # GEC -> edit rate (Apache-2.0)
    "textattack/roberta-base-CoLA",           # grammatical acceptability (MIT)
    "Qwen/Qwen3-1.7B",                        # LLM states + rubric score (Apache-2.0)
]
for m in MODELS:
    t = time.time()
    snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.bin", "*.tiktoken", "merges.txt", "vocab*"],
                      ignore_patterns=["*.msgpack", "*.h5", "*.ot", "flax*", "tf_*", "*onnx*"])
    print(f"{m}: {time.time() - t:.0f}s", flush=True)
print("all downloaded")
