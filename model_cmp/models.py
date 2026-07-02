from groq import Groq
import pandas as pd
import os
import time
import matplotlib.pyplot as plt

client = Groq(api_key=os.getenv("groq_api_key"))
conversation = [
    {
        "role": "system",
        "content": "You are a good data analyst. Explain datasets in simple language and provide business insights."
    }
]

MODELS = {
    "GPT-OSS-120B": "openai/gpt-oss-120b",
    "Llama-3.3-70B": "llama-3.3-70b-versatile",
    "Qwen-3-32B": "qwen/qwen3-32b"
}


def ask_model(model_name, messages):
    start=time.time()

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=1000,
        temperature=0.7
    )
    end=time.time()
    time_taken=round(end-start,2)
    return response.choices[0].message.content,time_taken


def compare_models(prompt):
    """
    Compare all models on the same prompt.
    """

    # Add the user's latest message
    conversation.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    results = {}

    for name, model in MODELS.items():
        answer,time_t = ask_model(model, conversation)
        results[name] ={"answer":answer,"time":time_t}
    conversation.append(
        {
            "role": "assistant",
            "content": results["GPT-OSS-120B"],
        }
    )

    return results

if __name__ == "__main__":

    df = pd.read_csv(
        r"C:\Users\nani3\Desktop\❤️‍\csv\sleep_cycle_productivity.csv"
    )

    summary = df.describe(include="all").to_string()

    prompt = f"""
Here are statistics from a dataset.
{summary}
Explain the data in a simple way.
Identify trends,
Identify anomalies,
Give business insights,
and give me the total information in only 10 lines
"""
responses = compare_models(prompt)
print("="*100)
print("summary of csv")
print("="*100)
for model, result in responses.items():
    print("=" * 100)
    print(model)
    print("=" * 100)
    print(f"Time Taken : {result['time']} seconds\n")
    print(result["answer"])
names=list(responses.keys())
time=[i['time'] for i in responses.values()]
plt.bar(names,time)
plt.xlabel('models')
plt.ylabel('time')
plt.title('models time')
plt.show()