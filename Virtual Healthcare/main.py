import os
import logging
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_community.tools.gmail.utils import (
    get_gmail_credentials,
    build_resource_service,
)
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# Set up API keys (ensure this is done securely, not hardcoded)
os.environ["SERPER_API_KEY"] = "59cced2f0003e5d0c055ebc9cc4676457bf25fdb"  # serper.dev API key
os.environ["OPENAI_API_KEY"] = "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"  # OpenAI API key


# Loading Tools
search_tool = SerperDevTool()

# Define the Agent (Virtual Health Care Intake Agent)
researcher = Agent(
    role='Senior Virtual Health Care Intake Agent',
    goal='Gather basic information about a patient and provide simple health advice.',
    backstory=( 
        "You are a Senior virtual Health care intake agent at a healthcare company. "
        "Your job is to gather basic information from the patient and provide a simple health report."
    ),
    verbose=True,
    allow_delegation=False,
    tools=[search_tool]
)

# Function to get Gmail service object using updated method
def get_gmail_service():
    """
    Build and return the Gmail service object by authenticating the credentials.
    """
    try:
        # Updated credential loading process with correct file paths
        credentials = get_gmail_credentials(
            token_file=os.getenv("GMAIL_TOKEN_PATH", "latest_ai_development/src/latest_ai_development/credentials/token.json"),  # use env variable
            scopes=["https://mail.google.com/"],
            client_secrets_file=os.getenv("GMAIL_CREDS_PATH", "latest_ai_development/src/latest_ai_development/credentials/credentials.json")  # use env variable
        )
        
        # Refresh the token if expired
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())  # Refresh the token if expired

    except Exception as e:
        logging.error(f"Error retrieving Gmail credentials: {e}")
        return None
    
    # Build Gmail service if credentials are valid
    service = build('gmail', 'v1', credentials=credentials)  # Builds the Gmail service
    return service

# Function to send an email using Gmail API directly
def send_email(patient_email, subject, body):
    """
    Function to send an email via Gmail using the Gmail API.
    """
    service = get_gmail_service()  # Get the Gmail service object
    if service is None:
        logging.error("Failed to authenticate with Gmail.")
        return

    try:
        # Prepare the email message
        message = create_message('me', patient_email, subject, body)

        # Send the email
        send_message(service, 'me', message)
        logging.info(f"Email sent successfully to {patient_email}")

    except Exception as e:
        logging.error(f"Error sending email: {e}")

# Helper function to create the message
def create_message(sender, to, subject, body):
    """
    Create a message for sending an email via Gmail API.
    """
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    import base64

    # Create a multipart message and set headers
    message = MIMEMultipart()
    message['to'] = to
    message['from'] = sender
    message['subject'] = subject

    # Add the body to the email
    msg = MIMEText(body)
    message.attach(msg)

    # Encode the message in base64 URL-safe encoding
    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

    return {'raw': raw_message}

# Helper function to send the message
def send_message(service, sender, message):
    """
    Send an email using the Gmail API.
    """
    try:
        # Send the email message
        service.users().messages().send(userId=sender, body=message).execute()
    except Exception as e:
        logging.error(f"Failed to send message: {e}")

# Define a task to process user inputs dynamically
def interact_with_patient():
    """Interactive chatbot for gathering basic health-related information"""
    print("Hello, I am your virtual health care assistant. How can I assist you today?")
    
    # Simulate the patient's email (you could collect this at the start of the conversation)
    patient_email = input("Please provide your email address: ")

    # Collect patient name and doctor name
    patient_name = input("Please provide your name: ")
    doctor_name = input("Please provide your doctor’s name: ")

    while True:
        # Get input from the user
        user_input = input("What happen to you?: ")

        if 'exit' in user_input.lower():
            print("Goodbye! Take care of your health.")
            break

        # Create a task based on the user's input
        task1 = Task(
            description=f"Based on the patient's statement: '{user_input}', provide a simple health summary.",
            expected_output='A simple health report summarizing the patient’s input.',
            agent=researcher,
            human_input=False  # No further human input required for this task
        )

        # Instantiate the crew and execute the task
        crew = Crew(
            agents=[researcher],
            tasks=[task1],
            verbose=True,
            memory=True,
            planning=True
        )

        # Get the results from the crew
        result = crew.kickoff()

        # Simplify the result (take only key points for a simple report)
        if hasattr(result, 'output'):
            result_text = result.output  # Extract the output text from the CrewOutput object
        else:
            result_text = str(result)  # Fallback if the result doesn't match expected output

        print("\n######################")
        print("Healthcare Advisor's Response:")
        print(result_text)
        print("\nFeel free to ask more questions or type 'exit' to end the chat.")

        # Construct a simple email body
        email_body = f"""
        Dear {patient_name},

        Dr. {doctor_name} has reviewed your health condition and here is a simple health report:

        {result_text}

        Best regards,
        Virtual Health Care Assistant
        """

        # After the conversation, send the simple health report directly to the patient's email
        send_email(patient_email, "Your Simple Health Report", email_body)

# Run the interactive chatbot
if __name__ == "__main__":
    # Set up logging for debugging
    logging.basicConfig(level=logging.DEBUG)

    interact_with_patient()
