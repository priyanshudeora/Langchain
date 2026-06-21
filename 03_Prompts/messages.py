from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
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


messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me about Langchain")
]

result = model.invoke(messages)
messages.append(AIMessage(content=result.content))
print(messages)