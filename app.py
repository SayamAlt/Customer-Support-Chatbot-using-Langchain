from langchain.document_loaders import JSONLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os, json, warnings, re
from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chat_models.openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

if "PINECONE_API_KEY" in st.secrets["secrets"]:
    pinecone_api_key = st.secrets["secrets"]["PINECONE_API_KEY"]
else:
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

if not pinecone_api_key:
    st.error("Missing Pinecone API credentials. Set them in the environment variables.")
    st.stop()

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)

# Define the Pinecone index name
index_name = "chatbot-index"

# Initialize the embeddings model
llm = ChatOpenAI(openai_api_key=api_key)

# Initialize the output parser
parser = StrOutputParser()

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

# Ensure that Pinecone index exists
if index_name not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,  # OpenAI embeddings dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Connect to the index
index = pc.Index(index_name)

# Store the embeddings in the Pinecone vector store
vector_store = LangchainPinecone.from_documents(splitted_docs,embeddings,index_name=index_name)

# Get the retriever from Pinecone vector store
retriever = vector_store.as_retriever()

# Initialize Conversation Buffer Memory to store conversational history
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, k=5)

# Define the Prompt Template
prompt = """
    You are an AI customer support assistant with a conversational tone, focused on being informative and efficient.

    Guidelines:

        - For casual questions (e.g., "How are you?"), respond naturally before offering assistance.
        - Use a casual tone for friendly users, and formal for professional ones.
        - Avoid sounding robotic—aim for natural, human-like phrasing.
        - Be concise, avoiding repetition or unnecessary details.
        - Show empathy when needed but avoid excessive apologies.
        - Maintain continuity in follow-up conversations.
        - If info is unavailable, suggest practical alternatives or next steps.

    Context from Previous Conversation:\n{context}\n
    Chat History:\n{chat_history}
    Answer user's query based on the guidelines, context, and chat history:\n{question}\n
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

def is_greeting(query):
    greeting_keywords = ["hello", "hi", "hey", "how are you", "good morning", "good afternoon", "good evening", "what's up"]
    query_lower = query.lower()
    for greeting in greeting_keywords:
        if re.search(r'\b' + re.escape(greeting) + r'\b', query_lower):
            return True
    return False

# Take user input
query = st.chat_input('Enter your question here: ')

if query:
    # Display the user's input in the chat window
    with st.chat_message("user"):
        st.markdown(query)

    # Check if the user's query is a greeting
    if is_greeting(query):
        template = """
            User has greeted with the following message: "{query}".
            Your task is to acknowledge the greeting in a friendly manner and offer assistance. 
            You should maintain a welcoming and approachable tone.
            Provide a response that feels natural, such as offering help or asking how you can assist further.
        """
        prompt = PromptTemplate(
            template=template,
            input_variables=['query']
        )
        chain = LLMChain(prompt=prompt,
                         llm=llm,
                         output_parser=parser)
        response = chain.run(query)

    else:
        # Store the conversation history for the last 5 user exchanges
        conversation_history = "\n".join([msg["content"] for msg in st.session_state.messages[-3:]])

        # Perform similarity search on the Pinecone vector store
        try:
            retrieved_docs = vector_store.similarity_search_with_score(query)            
        except PineconeApiAttributeError as e:
            warnings.warn(f"Error retrieving documents: {e}")
            retrieved_docs = []
        except Exception as e:
            warnings.warn(f"Error retrieving documents: {e}")
            retrieved_docs = []
        
        if not retrieved_docs:
            # If no relevant results are returned, respond with a default message
            response = "I'm sorry, I couldn't find any information related to that. Could you please clarify or ask something else?"

        # Extract text from page_content
        texts = []

        for doc, score in retrieved_docs:  # Unpacking document and similarity score
            try:
                # Parse the page_content as JSON
                parsed_content = json.loads(doc.page_content)
                texts.append(parsed_content["text"])  # Extract and store the text
            except (json.JSONDecodeError, KeyError):
                continue  # Skip if parsing fails

        # Combine all extracted text into a single string
        context = "\n".join(set(texts))  # Use set() to remove duplicates

        # Format the prompt with the user's query and the conversation history
        formatted_prompt = prompt.format(question=query,context=context,chat_history=conversation_history)

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
