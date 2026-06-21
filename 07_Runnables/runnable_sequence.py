from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence


model1 = ChatOllama(
    model="qwen3:4b",
    temperature=0
)

model2 = ChatOllama(
    model="gemma3:12b",
    temperature=0
)

template=PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=['topic']


)

template2=PromptTemplate(
    template='give expalination of following joke {text} ',
    input_variables=['text']
)

parser=StrOutputParser()

chain=RunnableSequence(template,model2,parser,template2,model1,parser)

print(chain.invoke('AI'))