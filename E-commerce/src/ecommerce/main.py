from crewai import Agent, Task, Crew, Process, LLM
from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource
import requests
from typing import Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import uuid
# Load environment variables from .env file
load_dotenv()

class ProductKnowledgeSource(BaseKnowledgeSource):
    """Knowledge source that fetches data from a product API."""
    
    api_endpoint: str = Field(description="API endpoint URL")
    
    def load_content(self) -> Dict[Any, str]:
        """Fetch and format product data from the remote API."""
        try:
            # Make a GET request to the API to fetch product data
            response = requests.get(self.api_endpoint)
            response.raise_for_status()  # Ensure the request was successful

            data = response.json()  # Parse the JSON response
            products = data.get('products', [])  # Extract the products from the response

            formatted_data = self._format_products(products)
            return {self.api_endpoint: formatted_data}
        except Exception as e:
            raise ValueError(f"Failed to fetch product data: {str(e)}")

    def _format_products(self, products: list) -> str:
        """Format products into readable text."""
        formatted = "Product Information:\n\n"
        for product in products:
            formatted += f"""
                Name: {product['title']}
                Price: {product['price']}
                Description: {product['description']}
                Category: {product['category']}
                URL: {product.get('meta', {}).get('qrCode', 'No URL available')}
                -------------------"""
        return formatted

    def add(self) -> None:
        """Process and store the product data."""
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
                for i in range(len(chunks))
            ]

        # Save documents (chunks) and metadata
        self.save_documents(metadata=chunks_metadata)

# Create knowledge source
product_knowledge = ProductKnowledgeSource(
    api_endpoint="https://dummyjson.com/products",  # Use remote API endpoint here
    
)

# Create specialized agent
product_analyst = Agent(
    role="Product Analyst",
    goal="Answer questions about product details and trends accurately and comprehensively",
    backstory="""You are a product analyst with expertise in various product categories,
    market trends, and consumer behavior. You excel at answering questions about products and providing detailed, accurate information.""",
    knowledge_sources=[product_knowledge],
    llm=LLM(model="gpt-4", temperature=0.0)
)

# Function to handle user questions dynamically
def handle_user_input(user_question: str):
    """Handles user input, creates a task, and returns the answer based on product knowledge."""
    
    # Create task for the user question
    analysis_task = Task(
        description=f"Answer this question about products: {user_question}",
        expected_output="A detailed answer based on the recent product data",
        agent=product_analyst
    )
    
    # Create and run the crew
    crew = Crew(
        agents=[product_analyst],
        tasks=[analysis_task],
        verbose=True,
        process=Process.sequential
    )
    
    # Get the answer for the user question
    result = crew.kickoff(
        inputs={"user_question": user_question}
    )
    
    return result

# Chatbot loop to allow user interaction
def chatbot_interaction():
    print("Welcome to the Product Knowledge Chatbot! Ask me anything about products.")
    
    while True:
        user_question = input("You: ")  # Ask the user for input
        if user_question.lower() in ['exit', 'quit', 'bye']:  # Allow the user to exit the chat
            print("Goodbye!")
            break
        
        # Get the answer based on the user's question
        answer = handle_user_input(user_question)
        
        # Display the answer
        print(f"Chatbot: {answer}\n")

# Start the chatbot interaction
chatbot_interaction()
