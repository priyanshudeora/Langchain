from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableSequence, RunnablePassthrough,
    RunnableBranch, RunnableLambda
)

prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)

model = ChatOllama(model="gemma3:12b", temperature=0)
parser = StrOutputParser()

report_gen_chain = prompt1 | model | parser

# Wrap raw string → {'text': ...} before feeding into prompt2
summarize_chain = RunnableLambda(lambda x: {'text': x}) | prompt2 | model | parser

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 300, summarize_chain),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

print(final_chain.invoke({'topic': 'Russia vs Ukraine'}))