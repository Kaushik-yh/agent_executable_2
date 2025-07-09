from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
import agents.agent as agent
# For Ollama model
from langchain_ollama import ChatOllama
# Other libs
import sys
import random

# Load chat-model
llm = ChatOllama(
    model = "gemma3:1b",
    temperature = 0.2,
    verbos=True  
)

app = FastAPI(title='My App',
              description='App to understand the sentiment and perform task accordingly',
              version='1.4.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=['GET','POST','PUT'],
    allow_headers=["*"]
)
my_agent = agent.MyAgent(llm)

class MyAppRequest(BaseModel):
    user_query: str

class MyAppResponce(BaseModel):
    message:Optional[str]
    category:Optional[str]
    sentiment:Optional[str]
    response:Optional[str]

class SessionStartResponse(BaseModel):
    session_id: Optional[str]
    message: Optional[str]

@app.post("/start_session", response_model=SessionStartResponse)
async def start_session():
    return SessionStartResponse(
        session_id = str(random.randint(0, 100)),
        message="Hello, I am your virtual assistant"
    )

@app.get("/chat", response_model=MyAppResponce)
async def chat(request: str) -> MyAppResponce:
    initial_state = {
        "message": [{"role":"user","content":request}] # request.user_query
    }
    #results = my_agent.workflow.invoke({"query": initial_state})
    results = my_agent.workflow.invoke(initial_state)

    #return str(results)
    return MyAppResponce(
        message=request,
        category=results.get("category"),
        sentiment=results.get("sentiment"),
        response=results.get("response")
    )  


@app.get("/")
async def root() -> Dict[str, Any]:
    return {'message':'welcome to the custom agent'}




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8004)

