
from llama_cpp import Llama

messages = [{"role": "system", "content": "What's 2+2?"},  # no system prompts for gemma, but it's okay
            {"role": "user", "content": "How are you today?"}]  

llm = Llama.from_pretrained(
    repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    filename="*Q2_K.gguf",
    verbose=False,
    chat_format="gemma",
)

x = llm.create_chat_completion(
    messages=messages,
)

print(x["choices"][0]["message"]["content"])
