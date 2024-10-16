from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def main():
    # Nome do modelo no Hugging Face
    model_name = "meta-llama/Llama-2-7b-chat-hf"

    # Carrega o tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Carrega o modelo
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Automaticamente mapeia o modelo para os dispositivos disponíveis (GPU/CPU)
        torch_dtype=torch.float16,  # Usa meia precisão para economizar memória
    )

    # Prepara o prompt
    prompt = "O que são LLMs?"

    # Tokeniza o prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Gera a resposta
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.7,
        do_sample=True,
    )

    # Decodifica e imprime a resposta
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response[len(prompt):].strip())

if __name__ == "__main__":
    main()
