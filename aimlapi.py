import openai
system_content = "You are a travel agent. Be descriptive and helpful."
user_content = "Tell me about San Francisco"
client = openai.OpenAI(
    api_key="d546f9f2469f46799b08a638d01fbd98",
    base_url="https://api.aimlapi.com/",
    )
chat_completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ],
    temperature=0.7,
    max_tokens=512,
)
response = chat_completion.choices[0].message.content
print("AI/ML API:\n", response)
