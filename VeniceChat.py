from openai import OpenAI

client = OpenAI(
    api_key="api_key",
    base_url="https://api.venice.ai/api/v1"
)

messages = [{"role": "system", "content": "Answer in English concisely"}]

while True:
    prompt = input("\nUser : ")
    if prompt.lower() == "exit":
        break
        
    messages.append({"role": "user", "content": prompt})
    
    stream = client.chat.completions.create(
        model="llama-3.2-3b",
        messages=messages,
        stream=True,
        temperature=0.7,
        max_tokens=500,
        extra_body={"venice_parameters": {"include_venice_system_prompt": False}}
    )
    
    print("\nAgent : ", end="", flush=True)
    full_response = ""
    
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            text = chunk.choices[0].delta.content
            print(text, end="", flush=True)
            full_response += text

    print("\n\n","="*50)
    messages.append({"role": "assistant", "content": full_response})