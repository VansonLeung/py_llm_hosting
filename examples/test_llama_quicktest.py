
from llama_cpp import Llama

messages = [{"role": "system", "content": "What's 2+2?"},  # no system prompts for gemma, but it's okay
            {"role": "user", "content": "How are you today?"}]  

llm = Llama.from_pretrained(
    repo_id="bartowski/gemma-2-9b-it-GGUF",
    filename="*Q6_K.gguf",
    verbose=False,
    chat_format="gemma",
)

x = llm.create_chat_completion(
    messages=messages,
)

print(x["choices"][0]["message"]["content"])
