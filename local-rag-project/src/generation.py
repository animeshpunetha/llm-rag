# This module structures the system prompt to prevent hallucinations and routes
# the compiled context directly to your local qwen2.5:7b model via Ollama.

from langchain_community.llms import Ollama
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
import config

def get_query_intent(query: str) -> str:
    """
    Acts as a Semantic Router. Classifies if the query is general chitchat 
    or if it requires searching the document database.
    """
    routing_instruction = (
        "You are an intent classification router. Classify the user's input into one of two categories:\n"
        "1. 'CHAT' - For greetings, pleasantries, or general conversational statements (e.g., 'hi', 'how are you', 'thanks').\n"
        "2. 'SEARCH' - For any questions asking for facts, details, or specific information.\n"
        "Output ONLY the exact word CHAT or SEARCH. Do not output anything else."
    )
    
    messages = [SystemMessage(content=routing_instruction), HumanMessage(content=query)]
    
    # We use temperature 0 for strict, predictable classification
    chat_model = ChatOllama(model=config.LLM_MODEL, temperature=0.0)
    response = chat_model.invoke(messages).content.strip().upper()
    
    # Fallback to SEARCH if the LLM says anything weird
    return "CHAT" if "CHAT" in response else "SEARCH"

def generate_chat_response(query: str, chat_history):
    """Generates a fast conversational response bypassing the vector database."""
    instruction = "You are a helpful and polite AI assistant. Keep your responses brief, friendly, and conversational."
    
    messages = [SystemMessage(content=instruction)]
    messages.extend(chat_history)
    messages.append(HumanMessage(content=query))
    
    # Temperature 0.4 allows for natural conversation
    chat_model = ChatOllama(model=config.LLM_MODEL, temperature=0.4)
    return chat_model.invoke(messages).content
    

def generate_answer(query: str, retrieved_docs, chat_history):
    # 1. Combine page contents from retrieved chunks into a single context string
    context_segments = []
    for i, doc in enumerate(retrieved_docs):
        source_page = doc.metadata.get("page", "Unknown")
        context_segments.append(f"--- Context Segment {i+1} (Page {source_page}) ---\n{doc.page_content}")

    compiled_context = "\n\n".join(context_segments)

    # 2. Construct a strict system prompt (STRICTLY BEHAVIOR ONLY - NO CONTEXT HERE)
    system_instruction = """You are a document question-answering system. 
Your task is to answer questions strictly from the provided document context.

Guidelines:
- Treat the document context as the only source of truth.
- Use chat history only to resolve references and follow-up questions.
- Do not use chat history as factual evidence.
- Do not rely on your pretrained knowledge.
- Do not fill gaps with assumptions.
- If evidence is missing, respond exactly: 'I cannot find the answer in the provided document.'
- If evidence is incomplete: answer only the supported portion and clearly state what is not available.
- If multiple context segments support the answer: synthesize them and do not invent connecting facts.
- If conflicting information appears: report the conflict and do not choose a side."""

    # 3. We inject the context directly into the final Human Message alongside the query
    contextualized_query = (
        f"DOCUMENT CONTEXT:\n{compiled_context}\n\n"
        f"USER QUESTION: {query}\n\n"
        "DETAILED ANSWER:"
    )

    # 4. Build the full message chain: Behavior -> Memory -> Facts + Question
    messages = [SystemMessage(content=system_instruction)]
    messages.extend(chat_history)  
    messages.append(HumanMessage(content=contextualized_query)) 

    # 5. Initialize connection with temperature=0.0 for strict extraction
    print(f"[Generation] Invoking local model {config.LLM_MODEL}...")
    chat_model = ChatOllama(model=config.LLM_MODEL, temperature=0.2)
    
    response = chat_model.invoke(messages)
    return response.content, retrieved_docs

def rewrite_query(query: str, chat_history) -> str:
    """
    Uses the LLM to rewrite a conversational follow-up question into a 
    standalone query optimized for vector database searching.
    """
    if not chat_history:
        return query  # If it's the first question, no need to rewrite
        
    rewrite_instruction = (
        "You are an expert search query optimizer. Given a conversation history and a follow-up question, "
        "rephrase the follow-up question into a standalone, complete search query that includes all necessary context. "
        "Do NOT answer the question. Only output the rephrased query string and nothing else."
    )
    
    messages = [SystemMessage(content=rewrite_instruction)]
    messages.extend(chat_history)
    messages.append(HumanMessage(content=f"Rephrase this follow-up question: {query}"))
    
    chat_model = ChatOllama(model=config.LLM_MODEL, temperature=0.0)
    response = chat_model.invoke(messages)
    
    clean_query = response.content.strip().strip('"').strip("'")
    print(f"[Query Router] Reformulated query to: '{clean_query}'")
    return clean_query
