from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os
from typing import Literal, TypedDict,Annotated,Optional
from pydantic import BaseModel,Field


load_dotenv()
os.environ['HF_HOME'] = 'C:/huggingface_cache'




model = ChatOllama(
    model="qwen3:4b",
    temperature=0
)



class Review(BaseModel):

    key_themes:list[str]=Field(description="Write down all the key themes discussed in the review in a list of strings")
    summery: str=Field(description="A brief summary of the review")
    sentiment:Literal["positive", "negative", "neutral"]=Field(description="The sentiment of the review, either positive, negative or neutral")
    pros: Optional[list[str]]=Field(default=None, description="A list of pros mentioned in the review, if any")
    cons: Optional[list[str]]=Field(default=None, description="A list of cons mentioned in the review, if any")
    name: Optional[str]=Field(default=None, description="The name of the product reviewer")

    

structured_output = model.with_structured_output(Review)


result= structured_output.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3
processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 50@@mAh battery easily
lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me
away is the 2@@MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x
actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with
bloatware-why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard
pill to swallow.

Pros:

Insanely powerful processor (great for gaming and productivity)
Stunning 2@@MP camera with incredible zoom capabilities

Long battery life with fast charging

S-Pen support is unique and useful I

Cons:

Bulky and heavy—not great for one-handed use
Bloatware still exists in One UI

Expensive compared to competitors

""")
print(result)

