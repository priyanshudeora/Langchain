from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("ai-2027.pdf")
docs = loader.load()



splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=""
)


# for doc in docs:
#     text = doc.page_content
#     chunks = splitter.split_text(text)
#     print(chunks)
#     print(len(chunks))
#     print("--------------------------------")   

chunks = splitter.split_documents(docs)
print(chunks)
print(len(chunks))