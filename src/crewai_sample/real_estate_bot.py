import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM
from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource
import requests
from typing import Dict, Any
from pydantic import Field
from dotenv import load_dotenv
import uuid

# Load environment variables from .env file
load_dotenv()

# Knowledge source class
class RealEstateKnowledgeSource(BaseKnowledgeSource):
    """Knowledge source that fetches data from a real estate API."""
    
    api_endpoint: str = Field(description="API endpoint URL")
    
    def load_content(self) -> Dict[Any, str]:
        """Fetch and format real estate data from the remote API."""
        try:
            response = requests.get(self.api_endpoint)
            response.raise_for_status()
            data = response.json()
            if not data.get("success", False):
                raise ValueError("API response indicates failure.")
            properties = data.get("data", [])
            if not properties:
                raise ValueError("No property data found in the API response.")
            return {self.api_endpoint: self._format_properties(properties)}
        except Exception as e:
            raise ValueError(f"Failed to fetch real estate data: {str(e)}")

    def _format_properties(self, properties: list) -> str:
        """Format real estate properties into readable text."""
        formatted = "Real Estate Listings:\n\n"
        for property in properties:
            formatted += f"""
                Title: {property['title']}
                Price: ${property['price']} per month
                Location: {property['location']}
                Bedrooms: {property['bedrooms']}
                Bathrooms: {property['bathrooms']}
                Property Type: {property['property_type']}
                Date Added: {property['date_added']}
                Images: {", ".join(property['images'])}
                -------------------\n
            """
        return formatted.strip()
    
    def add(self) -> None:
        """Process and store the real estate data."""
        content = self.load_content()
        for _, text in content.items():
            chunks = self._chunk_text(text)
            self.chunks.extend(chunks)
        
        # Save documents with metadata
        chunks_metadata = [
            {
                "chunk_id": str(uuid.uuid4()),
                "source": self.api_endpoint,
                "description": f"Chunk {i + 1} from API response"
            }
            for i in range(len(self.chunks))
        ]

        # Save documents (chunks) and metadata
        self.save_documents(metadata=chunks_metadata)

# Create knowledge source
real_estate_knowledge = RealEstateKnowledgeSource(
    api_endpoint="https://mocki.io/v1/504c8820-2957-495e-942d-b0bdec66b6d0",
)

# Create specialized agent
real_estate_agent = Agent(
    role="Real Estate Agent",
    goal="Answer questions about real estate properties and trends accurately and comprehensively",
    backstory="""You are a real estate agent with expertise in various property types, market trends, and neighborhood dynamics.
    You excel at answering questions about properties and providing detailed, accurate information.""",
    knowledge_sources=[real_estate_knowledge],
    llm=LLM(model="gpt-4o-mini", temperature=0.0)
)

# Streamlit app
st.title("🏠 Real Estate Knowledge Chatbot")
st.markdown("Ask me anything about real estate properties and trends!")

# Initialize session state for chat messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I assist you with real estate today?"}]

# Display chat messages from the session state
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle user input
if user_input := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Create a task for the user input
    task = Task(
        description=f"Answer this question about real estate: {user_input}",
        expected_output="A detailed answer based on the recent real estate data",
        agent=real_estate_agent,
    )
    
    # Create and run the crew
    crew = Crew(
        agents=[real_estate_agent],
        tasks=[task],
        process=Process.sequential
    )
    
    # Fetch the result
    try:
        result = crew.kickoff(inputs={"user_question": user_input})
        st.session_state.messages.append({"role": "assistant", "content": result})
        st.chat_message("assistant").write(result)
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        st.session_state.messages.append({"role": "assistant", "content": error_message})
        st.chat_message("assistant").write(error_message)
