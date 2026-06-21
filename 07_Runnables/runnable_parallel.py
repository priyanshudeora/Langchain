from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel


model1 = ChatOllama(
    model="qwen3:4b",
    temperature=0
)

model2 = ChatOllama(
    model="gemma3:12b",
    temperature=0
)


template1=PromptTemplate(
    template='generate a post for linkdin on {topic}',
    input_variables=['topic']
)

template2=PromptTemplate(
    template='generate a post for twitter on {topic}',
    input_variables=['topic']
)

parser=StrOutputParser()

chain=RunnableParallel(
    {
        'linkdin': RunnableSequence(template1,model1,parser),
        'twitter': RunnableSequence(template2,model2,parser)
    }
)

result=chain.invoke({'topic':"AI"})
print(result)