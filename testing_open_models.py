from togetherai import deepseek_70b, qwen_coder, deepseek_r1, llama_3b

def test_all_models():
    test_prompt = "Write a simple hello world program in Python."
    
    print("\nTesting DeepSeek-70B model:")
    print("-" * 50)
    result = deepseek_70b(test_prompt)
    print(result)
    
    print("\nTesting Qwen Coder model:")
    print("-" * 50) 
    result = qwen_coder(test_prompt)
    print(result)
    
    print("\nTesting DeepSeek-R1 model:")
    print("-" * 50)
    result = deepseek_r1(test_prompt)
    print(result)
    
    print("\nTesting Llama-3B model:")
    print("-" * 50)
    result = llama_3b(test_prompt)
    print(result)

if __name__ == "__main__":
    test_all_models()
