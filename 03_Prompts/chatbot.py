from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os

os.environ['HF_HOME'] = 'C:/huggingface_cache'

llm = HuggingFacePipeline.from_model_id(
    model_id='Qwen/Qwen3-4B',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)

print(llm.pipeline.model.device)
model = ChatHuggingFace(llm=llm)



chat_history = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me about Langchain")
]

while True:
    user_input = input("User: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() == "exit":
        break
    response = model.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    print("AI:", response.content)

print(chat_history)