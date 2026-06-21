from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

model = ChatOllama(
    model="qwen3:4b",
    temperature=1.6
)

parser=JsonOutputParser()   

template1=PromptTemplate(
    template='''Give me 5 facts about {topic} \n {format_instructions}''',
    input_variables=['topic'],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)
chain=template1 | model | parser
result=chain.invoke({'topic': 'Python programming'})
print(result)
print(type(result))

