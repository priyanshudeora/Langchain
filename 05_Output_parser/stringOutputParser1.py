from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

parser=StrOutputParser()


chain1=template1 | model | parser | template2 | model | parser

result=chain1.invoke({'topic':"Black Hole"})
print(result)


