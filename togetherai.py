# To use the functinos just import function_name from togetherai.py and call it with a prompt
from together import Together

client = Together(api_key="")

# Functions for different models
def deepseek_70b(prompt: str) -> str:
    """Run inference using DeepSeek-R1-Distill-Llama-70B-free model"""
    response_stream = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1302,
        temperature=0.7,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1,
        stop=["<｜end▁of▁sentence｜>"],
        stream=True
    )
    return _process_stream(response_stream)



def qwen_coder(prompt: str) -> str:
    """Run inference using Qwen2.5-Coder-32B-Instruct model"""
    response_stream = client.chat.completions.create(
        model="Qwen/Qwen2.5-Coder-32B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1302,
        temperature=0.7,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1,
        stream=True
    )
    return _process_stream(response_stream)

def deepseek_r1(prompt: str) -> str:
    """Run inference using DeepSeek-R1 model"""
    response_stream = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1302,
        temperature=0.7,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1,
        stream=True
    )
    return _process_stream(response_stream)

def llama_3b(prompt: str) -> str:
    """Run inference using Llama-3.2-3B-Instruct-Turbo model"""
    response_stream = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct-Turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1302,
        temperature=0.7,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1,
        stream=True
    )
    return _process_stream(response_stream)

def _process_stream(response_stream) -> str:
    """Helper function to process the response stream"""
    output = ""
    for token in response_stream:
        if hasattr(token, 'choices'):
            output += token.choices[0].delta.content
    return output



# ------------------------------------------------------------------------------------------------
# def infer(query: str) -> str:
    
#     # Here we're sending a simple conversation with one user message.
#     # (You can adjust the conversation history if needed.)
#     response_stream = client.chat.completions.create(
#         model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
#         messages=[
#             {
#                 "role": "user",
#                 "content": query
#             }
#         ],
#         max_tokens=1302,
#         temperature=0.7,
#         top_p=0.7,
#         top_k=50,
#         repetition_penalty=1,
#         stop=["<｜end▁of▁sentence｜>"],
#         _gl="1*1wnh5mi*_gcl_au*NjgwMzQxNzk0LjE3MzkwMzY0MjI.*_ga*MTk1MTU4Mzc4OS4xNzM5MDM2NDIy*_ga_BS43X21GZ2*MTczOTAzNjQyMi4xLjAuMTczOTAzNjQyMi4wLjAuMA..",
#         stream=True  # Using stream=True to receive tokens incrementally.
#     )
    
#     # Accumulate tokens from the stream into a single string.
#     output = ""
#     for token in response_stream:
#         if hasattr(token, 'choices'):
#             # Each token's content is available in token.choices[0].delta.content
#             output += token.choices[0].delta.content
            
#     return output
# if __name__ == "__main__":
#     print("Chat with the AI (type 'bye' to exit)")
#     while True:
#         query = input("\nYou: ").strip()
#         if query.lower() == "bye":
#             print("Goodbye!")
#             break
        
#         result = infer(query)
#         print("\nAI:", result)

# ------------------------------------------------------------------------------------------------

# Functions for different models
def deepseek_70b(prompt: str) -> str:
    """Run inference using DeepSeek-R1-Distill-Llama-70B-free model"""
    response_stream = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1302,
        temperature=0.7,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1,
        stop=["<｜end▁of▁sentence｜>"],
        stream=True
    )
    return _process_stream(response_stream)

def qwen_coder(prompt: str) -> str:
    """Run inference using Qwen2.5-Coder-32B-Instruct model"""
    response_stream = client.chat.completions.create(
        model="Qwen/Qwen2.5-Coder-32B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1302,
        temperature=0.7,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1,
        stream=True
    )
    return _process_stream(response_stream)

def deepseek_r1(prompt: str) -> str:
    """Run inference using DeepSeek-R1 model"""
    response_stream = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1302,
        temperature=0.7,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1,
        stream=True
    )
    return _process_stream(response_stream)

def llama_3b(prompt: str) -> str:
    """Run inference using Llama-3.2-3B-Instruct-Turbo model"""
    response_stream = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct-Turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1302,
        temperature=0.7,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1,
        stream=True
    )
    return _process_stream(response_stream)

def _process_stream(response_stream) -> str:
    """Helper function to process the response stream"""
    output = ""
    for token in response_stream:
        if hasattr(token, 'choices'):
            output += token.choices[0].delta.content
    return output
