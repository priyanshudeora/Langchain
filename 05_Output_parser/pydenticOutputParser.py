from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field


model = ChatOllama(
    model="qwen3:4b",
    temperature=1.6
)


class Person(BaseModel):
    name: str = Field(description="name of the person")
    age: int = Field(gt=18,description="age of the person")
    occupation: str = Field(description="occupation of the person")
    city: str = Field(description="city of the person")



parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate the name, age and city of a fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser
result=chain.invoke({'place':'Chinese'})
print(result)