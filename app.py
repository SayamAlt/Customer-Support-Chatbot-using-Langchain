from langchain.document_loaders import JSONLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os, json, warnings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models.openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import streamlit as st

# Load the environment variables
load_dotenv()

# Load the OpenAI API key
if "OPENAI_API_KEY" in st.secrets["secrets"]:
    api_key = st.secrets["secrets"]["OPENAI_API_KEY"]
else:
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error('Missing OpenAI API key. Please set it in the environment variables.')
    st.stop()

llm = ChatOpenAI(openai_api_key=api_key)

# Load the Customer Support FAQs JSON data
faq_file = "customer_support_faqs.json"

try:
    jq_schema = ".[] | {text: (.question + \" - \" + .answer)}"
    data = JSONLoader(file_path=faq_file, jq_schema=jq_schema, text_content=False)
    documents = data.load()
except (json.JSONDecodeError, FileNotFoundError):
    st.error("Error loading FAQ data. Ensure the JSON file is properly formatted and exists.")
    st.stop()

def save_to_dataset(question, answer):
    """
        Save user feedback to the dataset.

        Args:
            question (str): The user's question.
            answer (str): The user's answer.
    """
    feedback_entry = {"question": question, "answer": answer}
    
    try:
        with open(faq_file, "r+", encoding="utf-8") as f:
            data = json.load(f)
            data.append(feedback_entry)
            f.seek(0)
            json.dump(data, f, indent=4)
    except (json.JSONDecodeError, FileNotFoundError):
        with open(faq_file, "w", encoding="utf-8") as f:
            json.dump([feedback_entry], f, indent=4)

    st.success("User feedback saved successfully!")

# Initialize the session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Split the text into chunks
text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=20)
splitted_docs = text_splitter.split_documents(documents)

# Initialize the OpenAI embeddings
embeddings = OpenAIEmbeddings(api_key=api_key)

persist_directory = './customer_support_db'

# Load or create Chroma vector store
if not os.path.exists(persist_directory):
    chroma = Chroma.from_documents(
        documents=splitted_docs, embedding=embeddings, persist_directory=persist_directory
    )
    chroma.persist()
else:
    chroma = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# Get the retriever from Chroma vector store
retriever = chroma.as_retriever()

# Initialize Conversation Buffer Memory to store conversational history
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, k=5)

# Define the Prompt Template
prompt = """
    You are an AI-powered customer support assistant with a natural, conversational tone. Your goal is to engage in human-like interactions while staying informative and efficient.

    Guidelines:
    - If the user asks general questions (e.g., "How are you?"), respond naturally instead of redirecting immediately to assistance.  
    - Maintain a casual yet professional tone. Adapt based on user behavior (friendly if casual, formal if professional).  
    - Avoid sounding robotic. Use natural human-like phrasing.  
    - Be concise and solution-oriented. No unnecessary repetition.  
    - Express empathy when needed (e.g., concerns, complaints, delays) but avoid overuse of apologies.  
    - For follow-ups, ensure continuity instead of restarting every response from scratch.  
    - If information is unavailable, suggest practical alternatives or next steps.  

    Answer the following user query based on above guidelines:\n{question}\n
    Context from Previous Conversation:\n{context}\n
"""

# Define the Conversational Retrieval Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

st.title('AI-Powered Customer Support Chatbot')

# Display the historical chat messages
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Take user input
query = st.chat_input('Enter your question here: ')

if query:
    # Display the user's input in the chat window
    with st.chat_message("user"):
        st.markdown(query)

    # Store the conversation history for the last 5 user exchanges
    conversation_history = "\n".join([msg["content"] for msg in st.session_state.messages[-3:]])

    # Format the prompt with the user's query and the conversation history
    formatted_prompt = prompt.format(question=query,context=conversation_history)

    # Check if the user's query has already been answered
    if any(msg['content'].lower() == query.lower() for msg in st.session_state.messages):
        response = "It looks like we already discussed that. Is there anything else you'd like to know?"
    else:
        # Get a response from the Conversational Retrieval Chain
        response = qa_chain.run({'question': formatted_prompt})

    # Store the user's query in the chat history
    st.session_state.messages.append({'role': 'user', 'content': query})
    # Store LLM's response in the chat history
    st.session_state.messages.append({'role': 'assistant', 'content': response})

    # Display the LLM's response in the chat window
    with st.chat_message('assistant'):
        st.markdown(response)

    # Ask for user feedback about the LLM's response
    user_feedback = st.radio('Was this response helpful?', ('Yes', 'No'), index=None, key=f'feedback_{len(st.session_state.messages)}')

    # Save user feedback to the dataset if the response was not helpful
    if user_feedback:
        if user_feedback.strip().lower() == 'no':
            save_to_dataset(question, response)
            st.write("I'm sorry I couldn't help. Could you provide more details or clarify your question?")
        elif user_feedback.strip().lower() == 'yes':
            st.success("Thanks for your feedback! I'm glad I could help.")
