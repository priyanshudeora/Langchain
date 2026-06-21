from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(
    model="qwen3:4b",
    temperature=1.6
)

template = PromptTemplate(
    template='Generate a joke about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

chain = template | model | parser
result = chain.invoke({'topic':'IIT'})
print(result)

chain.get_graph().print_ascii()