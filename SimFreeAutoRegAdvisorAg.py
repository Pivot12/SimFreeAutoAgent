
import os
import re
import base64
import requests
import io
import streamlit as st
import pandas as pd
import logging
from typing import Dict, List, Any, Tuple
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import networkx as nx

# Import other libraries with try/except for better error handling
try:
    import PyPDF2
    from streamlit_mermaid import st_mermaid
    from groq import Groq
    from bs4 import BeautifulSoup
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception as e:
    st.error(f"Error importing libraries: {str(e)}")
    st.info("Some dependencies might be missing. Check your requirements.txt file.")

# Set up logging with a more robust approach
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "app.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DiagnosticLogger:
    """Handles structured diagnostic logging for the application."""
    
    def __init__(self, log_file_path="logs/diagnostic.log", github_logging=False, 
                 github_repo_owner=None, github_repo_name=None, github_token=None):
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        
        self.logger = logging.getLogger("diagnostic_logger")
        self.logger.setLevel(logging.INFO)
        
        # Create handler for file
        handler = RotatingFileHandler(log_file_path, maxBytes=10*1024*1024, backupCount=5)
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Set up GitHub logging if enabled
        self.github_logging = github_logging
        if github_logging and github_repo_owner and github_repo_name:
            self.github_logger = GitHubLogger(
                repo_owner=github_repo_owner,
                repo_name=github_repo_name,
                token=github_token
            )
        else:
            self.github_logging = False
    
    def log_session(self, user_id, query, market, accessed_documents, error=None, answer=None):
        """Log a complete session with structured data."""
        session_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        
        # Create structured log entry
        log_entry = {
            "session_id": session_id,
            "timestamp": timestamp,
            "user_id": user_id,
            "query": query,
            "market": market,
            "accessed_documents": accessed_documents,
            "error": error,
            "answer": answer
        }
        
        # Log as JSON string to file
        self.logger.info(json.dumps(log_entry))
        
        # Also log to GitHub if enabled
        if self.github_logging:
            try:
                self.github_logger.push_log(log_entry)
            except Exception as e:
                self.logger.error(f"Failed to log to GitHub: {str(e)}")
        
        return session_id
    
    def get_user_id(self, ip_address):
        """Generate a consistent but anonymized user ID from IP address."""
        # Hash the IP address to anonymize it
        return hashlib.sha256(ip_address.encode()).hexdigest()[:16]

class GitHubLogger:
    """Handles logging to a GitHub repository."""
    
    def __init__(self, repo_owner, repo_name, branch="main", log_file_path="logs/diagnostic_log.json", token=None):
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.branch = branch
        self.log_file_path = log_file_path
        self.token = token
        self.base_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}"
        self.headers = {
            "Accept": "application/vnd.github.v3+json"
        }
        if token:
            self.headers["Authorization"] = f"token {token}"
    
    def get_current_log(self):
        """Get the current log file from GitHub if it exists."""
        try:
            url = f"{self.base_url}/contents/{self.log_file_path}"
            response = requests.get(url, headers=self.headers, params={"ref": self.branch})
            
            if response.status_code == 200:
                content = response.json()
                file_content = base64.b64decode(content["content"]).decode("utf-8")
                sha = content["sha"]
                return json.loads(file_content), sha
            elif response.status_code == 404:
                # File doesn't exist yet
                return [], None
            else:
                logger.error(f"Failed to get log file: {response.status_code}, {response.text}")
                return [], None
        except Exception as e:
            logger.error(f"Error getting log file from GitHub: {str(e)}")
            return [], None
    
    def push_log(self, log_entry):
        """Push a new log entry to the GitHub repository."""
        current_logs, sha = self.get_current_log()
        
        # Add the new log entry
        current_logs.append(log_entry)
        
        # Convert to JSON
        content = json.dumps(current_logs, indent=2)
        
        # Encode content
        encoded_content = base64.b64encode(content.encode("utf-8")).decode("utf-8")
        
        # Prepare the commit data
        commit_data = {
            "message": f"Update diagnostic log: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "content": encoded_content,
            "branch": self.branch
        }
        
        if sha:
            commit_data["sha"] = sha
        
        # Push to GitHub
        try:
            url = f"{self.base_url}/contents/{self.log_file_path}"
            response = requests.put(url, headers=self.headers, json=commit_data)
            
            if response.status_code in [200, 201]:
                logger.info("Successfully pushed log to GitHub")
                return True
            else:
                logger.error(f"Failed to push log to GitHub: {response.status_code}, {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error pushing log to GitHub: {str(e)}")
            return False

# Function to get client IP address
def get_client_ip():
    """Get the client IP address for diagnostic logging."""
    # In a real deployment, you would access this from the request object
    # For a Streamlit app, this would need to be implemented based on your deployment
    # This is a placeholder - integrate with your hosting environment as needed
    return "127.0.0.1"
# ________________________________Core__________________________________________________________________
def download_and_process_pdfs(state):
    """Download PDFs and extract content, without simulating content."""
    logger.info("Downloading and processing PDFs...")
    pdf_urls = state.pdf_urls
    pdf_contents = {}
    
    successful_downloads = 0
    for title, url in pdf_urls:
        try:
            logger.info(f"Downloading PDF: {title} from {url}")
            
            # Skip example URLs that would lead to hallucination
            if "example.com" in url:
                logger.warning(f"Skipping example URL: {url}")
                continue
                
            response = requests.get(url, timeout=30)
            
            # Check if the response is valid and contains PDF content
            if response.status_code != 200:
                logger.warning(f"Failed to download PDF: {url}, status code: {response.status_code}")
                continue
                
            # Check content type to ensure it's a PDF
            content_type = response.headers.get('Content-Type', '').lower()
            if 'application/pdf' not in content_type and not url.lower().endswith('.pdf'):
                logger.warning(f"URL does not return PDF content: {url}, content type: {content_type}")
                continue
                
            pdf_file = io.BytesIO(response.content)
            
            # Read PDF content
            try:
                reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:  # Only add if text was successfully extracted
                        text += page_text + "\n"
                
                # Only add if we got actual content
                if text.strip():
                    pdf_contents[title] = text
                    successful_downloads += 1
                    logger.info(f"Successfully processed PDF: {title}, extracted {len(text)} characters")
                else:
                    logger.warning(f"No text could be extracted from PDF: {title}")
            except Exception as e:
                logger.error(f"Error reading PDF {title}: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error processing PDF {title}: {str(e)}")
    
    if successful_downloads == 0:
        logger.warning("No PDFs were successfully downloaded and processed")
    
    return {"pdf_contents": pdf_contents}

def extract_pdf_links(state):
    """Extract PDF links from the regulatory website without using example links."""
    logger.info("Extracting PDF links...")
    url = state.selected_url
    query = state.query
    
    try:
        logger.info(f"Fetching content from {url}")
        response = requests.get(url, timeout=30)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all links on the page
        links = soup.find_all('a')
        
        # Extract PDF links
        pdf_links = []
        for link in links:
            href = link.get('href')
            if href and href.endswith('.pdf'):
                full_url = href if href.startswith('http') else (url + href if not url.endswith('/') else url + '/' + href)
                if link.text:
                    pdf_links.append((link.text.strip(), full_url))
        
        logger.info(f"Found {len(pdf_links)} PDF links")
        
        if not pdf_links:
            logger.warning("No PDF links found on the regulatory website")
            return {"pdf_urls": []}
        
        # Use LLM to select relevant PDFs based on the query
        prompt = f"""
        Based on the user query: "{query}", select the most relevant PDF documents from the following list.
        Return the indices of the selected documents (0-based) as a comma-separated list.
        
        PDFs:
        {pd.DataFrame(pdf_links, columns=['Title', 'URL']).to_string()}
        
        Return only the indices as a comma-separated list, without any additional text.
        If none of the documents seem relevant to the query, return "NONE".
        """
        
        logger.info("Calling LLM to select relevant PDFs...")
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        
        # Extract indices from response
        indices_str = response.choices[0].message.content.strip()
        logger.info(f"LLM response for PDF selection: {indices_str}")
        
        if indices_str == "NONE":
            logger.warning("LLM determined no relevant PDFs for the query")
            return {"pdf_urls": []}
        
        indices = [int(idx.strip()) for idx in indices_str.split(',') if idx.strip().isdigit()]
        
        # Get selected PDFs
        selected_pdfs = [pdf_links[idx] for idx in indices if idx < len(pdf_links)]
        logger.info(f"Selected {len(selected_pdfs)} PDFs")
        
        return {"pdf_urls": selected_pdfs}
    
    except Exception as e:
        logger.error(f"Error extracting PDF links: {str(e)}")
        return {"pdf_urls": []}

def analyze_content(state):
    """Analyze PDF content and generate answer with factual verification."""
    logger.info("Analyzing content...")
    query = state.query
    pdf_contents = state.pdf_contents
    
    # Check if we have any PDF contents to analyze
    if not pdf_contents:
        logger.warning("No PDF contents available for analysis")
        return {"final_answer": "I'm sorry, but I couldn't find any relevant regulatory documents to answer your query. Please try a different query or select a specific market."}
    
    # Combine all contents
    combined_text = ""
    for title, content in pdf_contents.items():
        combined_text += f"--- Document: {title} ---\n{content}\n\n"
    
    # Split text into chunks to stay within token limits
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=30000,  # Adjust based on token limit
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(combined_text)
    logger.info(f"Split content into {len(chunks)} chunks")
    
    # Process chunks and collect insights with citations
    insights = []
    
    for i, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {i+1}/{len(chunks)}")
        prompt = f"""
        I'm analyzing automotive regulatory documents to answer a user's query.
        
        User query: {query}
        
        Document text (chunk {i+1}/{len(chunks)}):
        {chunk}
        
        For this text chunk, please:
        1. Extract key insights relevant to the query
        2. For each insight, provide the exact source document name and direct quotes that support it
        3. If you don't find any relevant information in this chunk, explicitly state "NO RELEVANT INFORMATION FOUND IN THIS CHUNK"
        
        Format each insight as:
        INSIGHT: [Your insight here]
        SOURCE: [Document name]
        EVIDENCE: "[Direct quote from document]"
        
        Be strict about only including insights with direct evidence from the documents.
        """
        
        try:
            logger.info(f"Calling LLM for chunk {i+1} analysis...")
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )
            chunk_insights = response.choices[0].message.content
            
            # Only add if there are actually insights found
            if "NO RELEVANT INFORMATION FOUND IN THIS CHUNK" not in chunk_insights:
                insights.append(chunk_insights)
            logger.info(f"Successfully analyzed chunk {i+1}")
        except Exception as e:
            logger.error(f"Error processing chunk {i+1}: {str(e)}")
    
    # If no insights found at all, return a "no answer" response
    if not insights:
        logger.warning("No relevant insights found in any document chunks")
        return {"final_answer": "I'm sorry, but I couldn't find any information in the regulatory documents that addresses your query. The documents I examined don't contain specific information about this topic."}
    
    # Combine insights and generate final answer with factual verification
    combined_insights = "\n\n".join(insights)
    
    prompt = f"""
    Based on the following insights extracted from automotive regulatory documents, provide a comprehensive answer to the user's query.
    
    User query: {query}
    
    Insights from documents (with sources and evidence):
    {combined_insights}
    
    Important instructions:
    1. Your answer MUST be derived ONLY from the document insights provided above
    2. Each statement in your answer must include a citation to the specific document source
    3. Do not make any claims or statements that aren't directly supported by the document evidence
    4. If the insights don't fully address the query, acknowledge the limitations of the available information
    5. If the insights don't address the query at all, respond with: "I apologize, but I couldn't find information that addresses your query in the regulatory documents I examined."
    6. Format citations as: [Source Document Name]
    
    Provide a well-structured, accurate, and factual answer focusing ONLY on what's present in the automotive regulations documents.
    """
    
    try:
        logger.info("Generating final answer...")
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000
        )
        final_answer = response.choices[0].message.content
        
        # Verification check
        if "I apologize, but I couldn't find information" in final_answer:
            logger.info("Response indicates no relevant information found")
        else:
            # Verify that the answer contains citations
            if "[" not in final_answer and "]" not in final_answer:
                logger.warning("Answer doesn't contain proper citations, likely hallucinating")
                final_answer = "I apologize, but I couldn't find specific information in the regulatory documents that addresses your query. Please try rephrasing your question or selecting a different market."
        
        logger.info("Final answer generated successfully")
    except Exception as e:
        logger.error(f"Error generating final answer: {str(e)}")
        final_answer = f"I apologize, but I encountered an error while processing your query. Please try again or rephrase your question."
    
    return {"final_answer": final_answer}


# ________________________________________________Simplified Agent______________________________________________________________________________________

def create_simple_agent():
    """Create a simplified agent without using LangGraph."""
    def process_query(query, market=None):
        logger.info("Starting simplified agent processing")
        results = {
            "query": query,
            "market": market,
            "selected_url": "",
            "pdf_urls": [],
            "pdf_contents": {},
            "final_answer": ""
        }
        
        # Step 1: Determine market if not provided
        if not market:
            logger.info("Determining market...")
            prompt = f"""
            Based on the following query, determine which automotive regulatory market the user is interested in (US, EU, China, India, or Australia).
            If the market is not clear, respond with "UNCLEAR".
            
            Query: {query}
            
            Return only the market name or "UNCLEAR" without any additional text.
            """
            
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10
                )
                market = response.choices[0].message.content.strip()
                results["market"] = market
            except Exception as e:
                logger.error(f"Error determining market: {str(e)}")
                results["market"] = "UNCLEAR"
                results["final_answer"] = "I apologize, but I encountered an error while processing your query. Please try again or specify a market explicitly."
                return results
        
        # Step 2: Select URL
        if results["market"] in REGULATORY_WEBSITES:
            results["selected_url"] = REGULATORY_WEBSITES[results["market"]]
        else:
            logger.warning(f"Market {results['market']} not found in regulatory websites")
            results["selected_url"] = "UNCLEAR"
            results["final_answer"] = "I apologize, but I couldn't identify a relevant regulatory market for your query. Please select a specific market (US, EU, China, India, or Australia) and try again."
            return results
        
        # Step 3: Extract PDF links without using example links
        try:
            url = results["selected_url"]
            logger.info(f"Fetching content from {url}")
            
            response = requests.get(url, timeout=30)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            links = soup.find_all('a')
            pdf_links = []
            
            for link in links:
                href = link.get('href')
                if href and href.endswith('.pdf'):
                    full_url = href if href.startswith('http') else (url + href if not url.endswith('/') else url + '/' + href)
                    if link.text:
                        pdf_links.append((link.text.strip(), full_url))
            
            if not pdf_links:
                logger.warning("No PDF links found on the regulatory website")
                results["final_answer"] = "I apologize, but I couldn't find any regulatory documents on the official website that address your query. Please try a different query or select a different market."
                return results
            
            # Select relevant PDFs
            prompt = f"""
            Based on the user query: "{query}", select the most relevant PDF documents from the following list.
            Return the indices of the selected documents (0-based) as a comma-separated list.
            
            PDFs:
            {pd.DataFrame(pdf_links, columns=['Title', 'URL']).to_string()}
            
            Return only the indices as a comma-separated list, without any additional text.
            If none of the documents seem relevant to the query, return "NONE".
            """
            
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100
            )
            
            indices_str = response.choices[0].message.content.strip()
            
            if indices_str == "NONE":
                logger.warning("LLM determined no relevant PDFs for the query")
                results["final_answer"] = "I apologize, but I couldn't find any regulatory documents that appear relevant to your query. Please try rephrasing your question or selecting a different market."
                return results
            
            indices = [int(idx.strip()) for idx in indices_str.split(',') if idx.strip().isdigit()]
            selected_pdfs = [pdf_links[idx] for idx in indices if idx < len(pdf_links)]
            
            results["pdf_urls"] = selected_pdfs
        except Exception as e:
            logger.error(f"Error extracting PDF links: {str(e)}")
            results["pdf_urls"] = []
            results["final_answer"] = f"I apologize, but I encountered an error while searching for relevant documents: {str(e)}. Please try again later."
            return results
        
        # Step 4: Download and process PDFs without simulating content
        pdf_contents = {}
        successful_downloads = 0
        
        for title, url in results["pdf_urls"]:
            try:
                logger.info(f"Downloading PDF: {title} from {url}")
                
                # Skip example URLs
                if "example.com" in url:
                    logger.warning(f"Skipping example URL: {url}")
                    continue
                
                response = requests.get(url, timeout=30)
                
                # Check if the response is valid
                if response.status_code != 200:
                    logger.warning(f"Failed to download PDF: {url}, status code: {response.status_code}")
                    continue
                    
                pdf_file = io.BytesIO(response.content)
                
                # Read PDF content
                reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:  # Only add if text was successfully extracted
                        text += page_text + "\n"
                
                # Only add if we got actual content
                if text.strip():
                    pdf_contents[title] = text
                    successful_downloads += 1
                    logger.info(f"Successfully processed PDF: {title}")
                else:
                    logger.warning(f"No text could be extracted from PDF: {title}")
            except Exception as e:
                logger.error(f"Error processing PDF {title}: {str(e)}")
        
        if successful_downloads == 0:
            logger.warning("No PDFs were successfully downloaded and processed")
            results["final_answer"] = "I apologize, but I couldn't successfully download or process any of the regulatory documents. Please try again later or with a different query."
            return results
        
        results["pdf_contents"] = pdf_contents
        
        # Step 5: Analyze content with factual verification
        insights = []
        combined_text = ""
        
        for title, content in results["pdf_contents"].items():
            combined_text += f"--- Document: {title} ---\n{content}\n\n"
        
        # Split into chunks if needed
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=30000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(combined_text)
        
        for i, chunk in enumerate(chunks):
            prompt = f"""
            I'm analyzing automotive regulatory documents to answer a user's query.
            
            User query: {query}
            
            Document text (chunk {i+1}/{len(chunks)}):
            {chunk}
            
            For this text chunk, please:
            1. Extract key insights relevant to the query
            2. For each insight, provide the exact source document name and direct quotes that support it
            3. If you don't find any relevant information in this chunk, explicitly state "NO RELEVANT INFORMATION FOUND IN THIS CHUNK"
            
            Format each insight as:
            INSIGHT: [Your insight here]
            SOURCE: [Document name]
            EVIDENCE: "[Direct quote from document]"
            
            Be strict about only including insights with direct evidence from the documents.
            """
            
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000
                )
                chunk_insights = response.choices[0].message.content
                
                # Only add if there are actually insights found
                if "NO RELEVANT INFORMATION FOUND IN THIS CHUNK" not in chunk_insights:
                    insights.append(chunk_insights)
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {str(e)}")
        
        # If no insights found at all, return a "no answer" response
        if not insights:
            logger.warning("No relevant insights found in any document chunks")
            results["final_answer"] = "I'm sorry, but I couldn't find any information in the regulatory documents that addresses your query. The documents I examined don't contain specific information about this topic."
            return results
        
        # Generate final answer with factual verification
        combined_insights = "\n\n".join(insights)
        
        prompt = f"""
        Based on the following insights extracted from automotive regulatory documents, provide a comprehensive answer to the user's query.
        
        User query: {query}
        
        Insights from documents (with sources and evidence):
        {combined_insights}
        
        Important instructions:
        1. Your answer MUST be derived ONLY from the document insights provided above
        2. Each statement in your answer must include a citation to the specific document source
        3. Do not make any claims or statements that aren't directly supported by the document evidence
        4. If the insights don't fully address the query, acknowledge the limitations of the available information
        5. If the insights don't address the query at all, respond with: "I apologize, but I couldn't find information that addresses your query in the regulatory documents I examined."
        6. Format citations as: [Source Document Name]
        
        Provide a well-structured, accurate, and factual answer focusing ONLY on what's present in the automotive regulations documents.
        """
        
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000
            )
            final_answer = response.choices[0].message.content
            
            # Verification check
            if "I apologize, but I couldn't find information" in final_answer:
                logger.info("Response indicates no relevant information found")
            else:
                # Verify that the answer contains citations
                if "[" not in final_answer and "]" not in final_answer:
                    logger.warning("Answer doesn't contain proper citations, likely hallucinating")
                    final_answer = "I apologize, but I couldn't find specific information in the regulatory documents that addresses your query. Please try rephrasing your question or selecting a different market."
            
            results["final_answer"] = final_answer
        except Exception as e:
            logger.error(f"Error generating final answer: {str(e)}")
            results["final_answer"] = f"I apologize, but I encountered an error while processing your query. Please try again or rephrase your question."
        
        return results

    return process_query
