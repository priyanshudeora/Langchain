from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableBranch,RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal


model=ChatOllama(
    model="gemma3:12b",
    temperature=0
)

class ReviewSentiment(BaseModel):
    sentiment: Literal['Positive', 'Negative'] = Field(description="The sentiment of the review")

parser=PydanticOutputParser(pydantic_object=ReviewSentiment)

template1=PromptTemplate(
    template='''Classify the sentiment of the following review.\n\nRespond with exactly one word:\nPositive\nNegative\n\nReview:\n{text} \n {format_instruction}''',
    input_variables=['text'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)



classifier=template1 | model | parser  
review='''I recently purchased the XYZ product and I am extremely satisfied with its performance. The build quality is excellent, and it works exactly as advertised. I would highly recommend this product to anyone in the market for something like this. Great value for the price!'''
# print(classifier.invoke({'text':review}))

template2=PromptTemplate(
    template='''Write an appropriate response to this positive review:\n{text}''',
    input_variables=['text']    
)
template3=PromptTemplate(
    template='''Write an appropriate response to this negative review:\n{text}''',
    input_variables=['text']
)
parser2=StrOutputParser()

branch_chain=RunnableBranch(
    (lambda x: x.sentiment == 'Positive', template2 | model | parser2),
    (lambda x: x.sentiment == 'Negative', template3 | model | parser2),
    RunnableLambda(lambda x: "Invalid sentiment")



)

chain=classifier | branch_chain

result=chain.invoke({'text': review})
print(result)

chain.get_graph().print_ascii()
