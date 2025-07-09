from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
# Load nodes.py
import re
from langchain_core.prompts import ChatPromptTemplate

# Define State Structure
#class RequestState(BaseModel):
#    query: Optional[str]

#class ResponceState(BaseModel):
#    query: Optional[str]
#    category: Optional[str]
#    sentiment: Optional[str]
#    response: Optional[str]

class WFState(TypedDict):
    query: Optional[str]
    category: Optional[str]
    sentiment: Optional[str]
    response: Optional[str]

class MyAgent:
    def __init__(self, llm):
        self.workflow = self.create_workflow()
        self.llm = llm

    # Define Node Function for identifying the category
    def categorize(self, state: str) -> Dict[str, Any]:
        """Categorize the customer query into Technical, Billing, or General."""
        query = state #.get('query')
        prompt = ChatPromptTemplate.from_template(
            "Categorize the following customer query into one of these categories: "
            "Technical, Billing, General. Query: {query}"
            "If unable to determine the right category, classify it as General, don't keep it null"
        )
        chain = prompt | self.llm
        category = chain.invoke({"query": query}).content
        selected_text = re.search(r"\*\*(.*?)\*\*", category)
        if selected_text:
            return {"category": selected_text.group(1)}
        else:
            return {"category": 'General'}

    # Define Node Function for analysing the sentiment
    def analyze_sentiment(self, state: str) -> Dict[str, Any]:
        query = state #.get('query')
        """Analyze the sentiment of the customer query as Positive, Neutral, or Negative."""
        prompt = ChatPromptTemplate.from_template(
            "Analyze the sentiment of the following customer query. "
            "Respond with either 'Positive', 'Neutral', or 'Negative'. Query: {query}"
        )
        chain = prompt | self.llm
        sentiment = chain.invoke({"query": query}).content
        return {"sentiment": sentiment}

    # Define Node Function for handel technical queries
    def handle_technical(self, state: str) -> Dict[str, Any]:
        """Provide a technical support response to the query."""
        query = state #.get('query')
        prompt = ChatPromptTemplate.from_template(
            "Provide a technical support response to the following query: {query}"
        )
        chain = prompt | self.llm
        response = chain.invoke({"query": query}).content
        return {"response": response}

    # Define Node Function for handel billing queries
    def handle_billing(self, state: str) -> Dict[str, Any]:
        """Provide a billing support response to the query."""
        query = state #.get('query')
        prompt = ChatPromptTemplate.from_template(
            "Provide a billing support response to the following query: {query}"
        )
        chain = prompt | self.llm
        response = chain.invoke({"query": query}).content
        return {"response": response}

    # Define Node Function for handel general queries
    def handle_general(self, state: str) -> Dict[str, Any]:
        """Provide a general support response to the query."""
        query = state #.get('query')
        prompt = ChatPromptTemplate.from_template(
            "Provide a general support response to the following query: {query}"
        )
        chain = prompt | self.llm
        response = chain.invoke({"query": query}).content
        return {"response": response}

    # Define Node Function for escalate to human
    def escalate(self, state: str) -> Dict[str, Any]:
        """Escalate the query to a human agent due to negative sentiment."""
        return {"response": "This query has been escalated to a human agent due to its negative sentiment."}

    # Define Node Function for route the workflow based on the sentiment
    def route_query(self, state: str) -> str:
        """Route the query based on its sentiment and category."""
        sentiment = state.get('sentiment')
        category = state.get('sentiment')
        if sentiment == "Negative":
            return "escalate"
        elif category == "Technical":
            return "handle_technical"
        elif category == "Billing":
            return "handle_billing"
        else:
            return "handle_general"
    
    def create_workflow(self) -> StateGraph:
        # Create the graph
        workflow = StateGraph(WFState)

        # Add nodes
        workflow.add_node("categorize", self.categorize)
        workflow.add_node("analyze_sentiment", self.analyze_sentiment)
        workflow.add_node("handle_technical", self.handle_technical)
        workflow.add_node("handle_billing", self.handle_billing)
        workflow.add_node("handle_general", self.handle_general)
        workflow.add_node("escalate", self.escalate)

        # Add edges
        workflow.add_edge("categorize", "analyze_sentiment")
        workflow.add_conditional_edges(
            "analyze_sentiment",
            self.route_query,
            {
                "handle_technical": "handle_technical",
                "handle_billing": "handle_billing",
                "handle_general": "handle_general",
                "escalate": "escalate"
            }
        )
        workflow.add_edge("handle_technical", END)
        workflow.add_edge("handle_billing", END)
        workflow.add_edge("handle_general", END)
        workflow.add_edge("escalate", END)

        # Set entry point
        workflow.set_entry_point("categorize")

        # Compile the graph
        return workflow.compile()
    
    def run(self, query: str) -> Dict[str, str]:
        initial_state = {
            'message':[{'role':'user', 'content': query}]
        }
        final_state = self.workflow.invoke(initial_state)
        return final_state