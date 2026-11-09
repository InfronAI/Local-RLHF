"""Build a small random local model for offline correctness tests only."""

from pathlib import Path


def make_tiny_model(directory):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    words = ["[PAD]", "[EOS]", "[UNK]", "user:", "assistant:", "system:",
             "What", "is", "one", "plus", "two", "?", "three", "four", "good", "bad",
             "Hello", "world", "Explain", "code", "Test", "the", "answer", "yes", "no"]
    tokenizer = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]",
                                   eos_token="[EOS]", unk_token="[UNK]", model_max_length=256)
    fast.chat_template = "{% for m in messages %}{{ m['role'] + ': ' + m['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
    fast.save_pretrained(directory)
    torch.manual_seed(7)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=len(words), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
        bos_token_id=None, eos_token_id=1, pad_token_id=0, tie_word_embeddings=True,
    ))
    model.save_pretrained(directory)
    return str(directory)
