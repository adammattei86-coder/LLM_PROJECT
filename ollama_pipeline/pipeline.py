import ollama

article = """
Artificial intelligence is rapidly transforming education.
Many schools are introducing AI-powered tutoring systems
to help students learn more efficiently.
Experts believe personalized learning will become increasingly common.
"""

# ZERO SHOT
zero_shot_prompt = f"""
Summarize this article:

{article}
"""

# FEW SHOT
few_shot_prompt = f"""
Article:
The weather is very hot today with temperatures reaching 35 degrees.

Summary:
Very high temperatures are expected today.

Article:
{article}

Summary:
"""

# SYSTEM PROMPT
system_prompt = """
You are an expert news summarizer.
Generate concise and professional summaries.
"""

print("\n===== ZERO SHOT =====\n")

response = ollama.chat(
    model="llama3",
    messages=[
        {
            "role": "user",
            "content": zero_shot_prompt
        }
    ]
)

print(response["message"]["content"])

print("\n===== FEW SHOT =====\n")

response = ollama.chat(
    model="llama3",
    messages=[
        {
            "role": "user",
            "content": few_shot_prompt
        }
    ]
)

print(response["message"]["content"])

print("\n===== SYSTEM PROMPT =====\n")

response = ollama.chat(
    model="llama3",
    messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": article
        }
    ]
)

print(response["message"]["content"])