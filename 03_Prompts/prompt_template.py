from langchain_core.prompts import PromptTemplate

# detailed way
template2 = PromptTemplate(
    template='Greet this person in 5 languages. The name of the person is ',
    input_variables=["paper_input",
        "style_input",
        "length_input"]
)
