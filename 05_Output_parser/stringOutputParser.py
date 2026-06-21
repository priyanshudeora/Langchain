from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

model = ChatOllama(
    model="qwen3:4b",
    temperature=0
)

template1=PromptTemplate(
    template='''Write a detailed on {topic}''',
    input_variables=['topic']
)


template2=PromptTemplate(
    template='''Write a 5 line summery of following text. /n {text}''',
    input_variables=['text']
)

prompt1=template1.format(topic="Black Hole")
result=model.invoke(prompt1)

prompt2=template2.format(text=result.content)
result2=model.invoke(prompt2)
print(result2.content)

