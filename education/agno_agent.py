from agno.agent import Agent
from agno.models.google import Gemini
import fitz  # PyMuPDF
import os

# Initialize the model
model = Gemini(api_key="Q5FTyYuGJiCOZLh7lFXoEsf8ytRuRyHgDMZU1xX7")

# Create the agent with summarization instructions
agent = Agent(
    model=model,
    instructions=["rephrase the following text without removing any ideas or dates."],
    markdown=True
)

for i in range(1,10):
    with open(f"document\\chapters\\chapter_{i}.txt", 'r', encoding='utf-8') as f:
        content = f.read()
    result = agent.run(content)

    with open(f"Agno_cohere\\chapter{i}.txt", 'w', encoding='utf-8') as f_out:
        f_out.write(result.content)
        print(f"rephrased chapter{i}")