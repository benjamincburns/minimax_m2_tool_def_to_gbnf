import os

from transformers import AutoTokenizer

cache_dir = os.path.join(os.path.dirname(__file__), "config")

minimax_tokenizer = AutoTokenizer.from_pretrained(
    "MiniMaxAI/MiniMax-M2.5",
    local_files_only=True,
    cache_dir=cache_dir,
)
