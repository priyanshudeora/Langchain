from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers.structured import (
    StructuredOutputParser,
    ResponseSchema
)

model = ChatOllama(
    model="qwen3:4b",
    temperature=0
)

schema=[
    ResponseSchema(name='fact_1', description='The  fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='The  fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='The  fact 3 about the topic'),
    ResponseSchema(name='fact_4', description='The  fact 4 about the topic'),
    ResponseSchema(name='fact_5', description='The  fact 5 about the topic'),
]

parser=StructuredOutputParser.from_response_schemas(schema)


template1=PromptTemplate(
    template='''Give me 5 facts about {topic} \n {format_instructions}''',
    input_variables=['topic'],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)

chain=template1 | model | parser
result=chain.invoke({'topic': 'Black Hole'})
print(result)