from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("ai-2027.pdf")
docs = loader.load()

print(len(docs))