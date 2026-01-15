import openai
from openai import OpenAI

def setup_openai_client(api_key_path='OAI_CONFIG_LIST'):
    with open(api_key_path) as f:
        api_key = f.read().strip()
    return openai.OpenAI(api_key=api_key)


client = setup_openai_client()
llm_model = "gpt-4o" 

# Function to get response from OpenAI LLM
def oai_get_response(msg, temp, top_p, seed=None):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": msg,
            }
        ],
        model=llm_model,
        temperature=temp,
        top_p=top_p,
    )

    reply_text = chat_completion.choices[0].message.content
    return reply_text