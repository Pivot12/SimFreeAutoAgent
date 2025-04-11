import os
import re
import base64
import requests
import io
import json
import datetime
import uuid
import hashlib
import logging
from logging.handlers import RotatingFileHandler
import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Tuple
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import networkx as nx

# Set page configuration early to avoid StreamlitAPIException
st.set_page_config(page_title="Automotive Regulations AI Agent", layout="wide")

# Create logs directory
os.makedirs("logs", exist_ok=True)

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add file handler
file_handler = RotatingFileHandler("logs/app.log", maxBytes=10*1024*1024, backupCount=5)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Display initial loading message
with st.spinner("Loading application dependencies..."):
    # Import optional dependencies with error handling
    try:
        import PyPDF2
        from bs4 import BeautifulSoup
        from groq import Groq
    except ImportError as e:
        st.error(f"Failed to import required dependencies: {str(e)}")
        logger.error(f"Import error: {str(e)}")
        st.stop()
# Diagnostic logger for structured logging
class DiagnosticLogger:
    """Handles structured diagnostic logging for the application."""
    
    def __init__(self, log_file_path="logs/diagnostic.json"):
        self.log_file_path = log_file_path
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        
        # Create file if it doesn't exist
        if not os.path.exists(log_file_path):
            with open(log_file_path, 'w') as f:
                f.write(json.dumps([]))
    
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
        
        # Read existing logs
        try:
            with open(self.log_file_path, 'r') as f:
                try:
                    logs = json.load(f)
                except json.JSONDecodeError:
                    logs = []
        except FileNotFoundError:
            logs = []
        
        # Append new log
        logs.append(log_entry)
        
        # Write back to file
        with open(self.log_file_path, 'w') as f:
            json.dump(logs, f, indent=2)
        
        return session_id
    
    def get_user_id(self, ip_address):
        """Generate a consistent but anonymized user ID from IP address."""
        # Hash the IP address to anonymize it
        return hashlib.sha256(ip_address.encode()).hexdigest()[:16]

# Github logging is optional and configured separately if needed
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
            "message": f"Update diagnostic log: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
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

# Diagram Image Creation
def create_diagram_image():
    """Create a polished diagram image using NetworkX and Matplotlib"""
    # Create a graph
    G = nx.DiGraph()
    
    # Node positions for better spacing and no overlaps
    nodes = {
        "User Input": {"pos": (0, 2)},
        "Market & Source Detection": {"pos": (0, 0)},
        "Process Query": {"pos": (2, 1)},
        "Initialize Agent": {"pos": (4, 1)},
        "Processing Pipeline": {"pos": (6, 1)},
        "Document Analysis": {"pos": (8, 1)},
        "Generate Answer": {"pos": (10, 1)},
        "Llama Gen-AI LLM": {"pos": (7, -1.5)},
        "PDF Processing": {"pos": (8, -0.5)},
        "Error Handling": {"pos": (5, -1.5)}
    }
    
    # Add all nodes
    for node, attrs in nodes.items():
        G.add_node(node, **attrs)
        
    # Enhanced node colors with better contrast
    node_colors = {
        "User Input": "#b8e0a1",          # Deeper green
        "Market & Source Detection": "#b8e0a1",     # Deeper green
        "Process Query": "#b8e0a1",        # Deeper green
        "Initialize Agent": "#a7c8f7",     # Deeper blue
        "Processing Pipeline": "#a7c8f7",  # Deeper blue
        "Document Analysis": "#a7c8f7",    # Deeper blue
        "Generate Answer": "#a7c8f7",      # Deeper blue
        "Llama Gen-AI LLM": "#f9c5a1",         # Deeper peach
        "PDF Processing": "#a7c8f7",       # Deeper blue
        "Error Handling": "#f9b0b0"        # Deeper red
    }
    
    # Edges
    edges = [
        ("User Input", "Process Query"),
        ("Market & Source Detection", "Process Query"),
        ("Process Query", "Initialize Agent"),
        ("Initialize Agent", "Processing Pipeline"),
        ("Processing Pipeline", "Document Analysis"),
        ("Document Analysis", "Generate Answer"),
        ("Document Analysis", "PDF Processing"),
    ]
    
    # Special edges to avoid label overlaps
    special_edges = [
        ("Llama Gen-AI LLM", "Process Query"),
        ("Llama Gen-AI LLM", "Processing Pipeline"),
        ("Llama Gen-AI LLM", "Document Analysis"),
        ("Llama Gen-AI LLM", "Generate Answer"),
        ("Error Handling", "Processing Pipeline"),
        ("Error Handling", "Document Analysis")
    ]
    
    G.add_edges_from(edges)
    G.add_edges_from(special_edges)
    
    # Figure with a white background
    plt.figure(figsize=(12, 7), facecolor='white')
    
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw Nodes
    for node, color in node_colors.items():
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=color,
                               node_size=3000, edgecolors='black', linewidths=2)
    
    # Draw regular edges (solid lines)
    nx.draw_networkx_edges(G, pos, edgelist=edges, arrows=True, arrowsize=20,
                          width=2, edge_color='black', connectionstyle='arc3,rad=0.0')
    
    # Draw special edges (dashed lines) with curved paths to avoid nodes
    edge_styles = {
        ("Llama Gen-AI LLM", "Process Query"): {'rad': 0.3, 'style': 'dashed', 'color': 'gray'},
        ("Llama Gen-AI LLM", "Processing Pipeline"): {'rad': 0.3, 'style': 'dashed', 'color': 'gray'},
        ("Llama Gen-AI LLM", "Document Analysis"): {'rad': 0.2, 'style': 'dashed', 'color': 'gray'},
        ("Llama Gen-AI LLM", "Generate Answer"): {'rad': 0.4, 'style': 'dashed', 'color': 'gray'},
        ("Error Handling", "Processing Pipeline"): {'rad': 0.3, 'style': 'dashed', 'color': 'gray'},
        ("Error Handling", "Document Analysis"): {'rad': 0.4, 'style': 'dashed', 'color': 'gray'}
    }
    
    for edge, style in edge_styles.items():
        nx.draw_networkx_edges(G, pos, edgelist=[edge], arrows=True, arrowsize=20,
                              width=2, edge_color=style['color'], style=style['style'],
                              connectionstyle=f'arc3,rad={style["rad"]}')
    
    # Draw node labels
    text_items = {}
    for node, (x, y) in pos.items():
        text_items[node] = plt.text(x, y, node,
                                   fontsize=11,
                                   fontweight='bold',
                                   ha='center', va='center',
                                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.5'),
                                   zorder=3)  # Higher zorder to be on top
    
    # Add a legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=node_colors["User Input"], 
                   markersize=15, label='User Interface'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=node_colors["Initialize Agent"], 
                   markersize=15, label='Processing Components'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=node_colors["Llama Gen-AI LLM"], 
                   markersize=15, label='External API'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=node_colors["Error Handling"], 
                   markersize=15, label='Error Handling'),
        plt.Line2D([0], [0], color='black', lw=2, label='Direct Flow'),
        plt.Line2D([0], [0], color='gray', lw=2, linestyle='--', label='Support Services')
    ]
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05),
               ncol=3, frameon=True, facecolor='white', edgecolor='lightgray')
    
    # Remove axes and add a title
    plt.axis('off')
    plt.title('Automotive Regulations AI Agent Architecture', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # Save the plot to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    plt.close()
    
    # Create image from buffer
    image = Image.open(buffer)
    return image


# Function to get base64 encoded image for embedded display
def get_image_base64(image):
   buffered = BytesIO()
   image.save(buffered, format="PNG")
   img_str = base64.b64encode(buffered.getvalue()).decode()
   return img_str

# Function to get client IP address
def get_client_ip():
    """Get the client IP address for diagnostic logging."""
    try:
        # In Streamlit, we can't directly access client IP
        # This is a placeholder - in production you would need to implement
        # based on your hosting environment
        return "127.0.0.1"
    except Exception as e:
        logger.error(f"Error getting client IP: {str(e)}")
        return "unknown"

# Initialize Groq client safely
def initialize_groq_client():
    """Initialize the Groq client with proper error handling."""
    try:
        # Try to get API key from environment variables
        api_key = os.environ.get("GROQ_API_KEY")
        
        # Try to get from Streamlit secrets if available
        if not api_key and hasattr(st, 'secrets'):
            try:
                api_key = st.secrets.get("GROQ_API_KEY", "")
            except Exception as e:
                logger.warning(f"Could not access Streamlit secrets: {str(e)}")
        
        # Fallback to hardcoded key (only for development)
        if not api_key:
            api_key = "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"
            logger.warning("Using hardcoded API key - not recommended for production")
        
        if not api_key:
            logger.error("No Groq API key found")
            return None
            
        return Groq(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {str(e)}")
        return None

REGULATORY_WEBSITES = {
    "Global & Regional Authorities UNECE" : "www.unece.org/trans/main/wp29/wp29regs.html​",
    
    "EU European Commission" : "ec.europa.eu/transport/home_en​",
    
    "European Automobile Manufacturers' Association (ACEA)" : "ACEA Regulatory Guide 2023​",
    
    "International Organization for Standardization (ISO) – Road Vehicles" : "www.iso.org/committee/45306.html​",
    
    "International Electrotechnical Commission (IEC) – Road Vehicles" : "www.iec.ch/standardsdev/publications/standards.htm​",
    
    "🇺🇸 United States - National Highway Traffic Safety Administration (NHTSA)" : "www.nhtsa.gov​",
    
    "US Environmental Protection Agency (EPA) – Vehicle Regulations" : "www.epa.gov/vehicle-and-engine-certification​",
    
    "European Free Trade Association (EFTA) – Vehicle Regulations" : "www.efta.int/eea/eea-legal-order/transport​",
    
    "🇯🇵 Japan Ministry of Land, Infrastructure, Transport and Tourism (MLIT)" : "www.mlit.go.jp/en/road/index.html​",
    
    "🇨🇳 China Ministry of Industry and Information Technology (MIIT)" : "www.miit.gov.cn/​",
    
    "🇮🇳 India Automotive Research Association of India (ARAI)" : "www.araiindia.com​",
    
    "Central Motor Vehicle Rules (CMVR)" : "www.morth.nic.in​",
    
    "🇨🇦 Canada Transport Canada – Motor Vehicle Safety" : "tc.canada.ca/en/road-transportation/motor-vehicle-safety​",
    
    "🇦🇺 Australia Vehicle Standards" : "www.infrastructure.gov.au/vehicles/vehicle-standards​",
    
    "🇧🇷 Brazil National Institute of Metrology, Quality and Technology (INMETRO)" : "www.gov.br/inmetro/pt-br​",
    
    "🇰🇷 South Korea Ministry of Land, Infrastructure and Transport (MOLIT)" : "www.molit.go.kr/english/​",
    
    "🇷🇺 Russia Federal Road Agency (Rosavtodor)" : "www.rosavtodor.ru/en/​",
    
    "🇲🇽 Mexico Secretariat of Communications and Transportation (SCT)" : "www.gob.mx/sct​",
    
    "🇿🇦 South Africa National Regulator for Compulsory Specifications (NRCS)" : "www.nrcs.org.za/​",
    
    "🇦🇷 Argentina National Road Safety Agency (ANSV)" : "www.ansv.gob.ar/​",
    
    "🇬🇧 United Kingdom Department for Transport (DfT)" : "www.gov.uk/government/organisations/department-for-transport",
}

# Create a mapping of countries/regions to their respective sources
def create_market_to_sources_mapping():
    """Create a mapping of markets to their respective regulatory sources."""
    market_to_sources = {}
    
    # Define market identification patterns
    market_patterns = {
        "Global": ["Global", "International", "UNECE", "ISO", "IEC"],
        "US": ["🇺🇸", "US", "United States", "NHTSA", "EPA"],
        "EU": ["🇪🇺", "EU", "European", "ACEA", "EFTA"],
        "UK": ["🇬🇧", "UK", "United Kingdom", "DfT"],
        "China": ["🇨🇳", "China", "MIIT"],
        "India": ["🇮🇳", "India", "ARAI", "CMVR"],
        "Japan": ["🇯🇵", "Japan", "MLIT"],
        "Canada": ["🇨🇦", "Canada"],
        "Australia": ["🇦🇺", "Australia"],
        "Brazil": ["🇧🇷", "Brazil", "INMETRO"],
        "South Korea": ["🇰🇷", "Korea", "MOLIT"],
        "Russia": ["🇷🇺", "Russia", "Rosavtodor"],
        "Mexico": ["🇲🇽", "Mexico", "SCT"],
        "South Africa": ["🇿🇦", "South Africa", "NRCS"],
        "Argentina": ["🇦🇷", "Argentina", "ANSV"]
    }
    
    # Categorize each source by market
    for source, url in REGULATORY_WEBSITES.items():
        assigned = False
        for market, patterns in market_patterns.items():
            if any(pattern in source for pattern in patterns):
                if market not in market_to_sources:
                    market_to_sources[market] = []
                market_to_sources[market].append(source)
                assigned = True
                break
        
        # If not assigned to any specific market, put in "Other"
        if not assigned:
            if "Other" not in market_to_sources:
                market_to_sources["Other"] = []
            market_to_sources["Other"].append(source)
    
    return market_to_sources

# Define state operations
def get_market_and_source(query, client):
    """Determine which market and regulatory source to use based on the query."""
    logger.info("Starting market and source determination...")
    
    # Create a formatted list of all available regulatory sources
    source_list = ""
    for source, url in REGULATORY_WEBSITES.items():
        source_list += f"- {source}: {url}\n"
    
    prompt = f"""
    Based on the following query about automotive regulations, determine:
    1. Which market (country/region) the user is interested in
    2. Which regulatory source would be most relevant to answer their query

    User query: {query}
    
    Available regulatory sources:
    {source_list}
    
    Respond in the following format:
    MARKET: [market name or "UNCLEAR"]
    SOURCE: [exact name of the most relevant regulatory source from the list above or "NONE" if unclear]
    
    If the market is unclear, respond with:
    MARKET: UNCLEAR
    SOURCE: NONE
    """
    
    try:
        # Call LLM to determine market and source
        logger.info("Calling LLM to determine market and source...")
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        
        result_text = response.choices[0].message.content.strip()
        logger.info(f"LLM response: {result_text}")
        
        # Parse the response
        market = "UNCLEAR"
        source = "NONE"
        
        lines = result_text.split('\n')
        for line in lines:
            if line.startswith("MARKET:"):
                market = line.replace("MARKET:", "").strip()
            elif line.startswith("SOURCE:"):
                source = line.replace("SOURCE:", "").strip()
        
        # Validate source is in our list
        if source != "NONE" and source not in REGULATORY_WEBSITES:
            # Try to find a close match
            for key in REGULATORY_WEBSITES.keys():
                if source in key or key in source:
                    source = key
                    break
            else:
                logger.warning(f"LLM returned invalid source: {source}")
                source = "NONE"
                
        logger.info(f"Market determined: {market}, Source: {source}")
        return market, source
    except Exception as e:
        logger.error(f"Error determining market and source: {str(e)}")
        return "UNCLEAR", "NONE"

def extract_pdf_links(url, query, client):
    """Extract PDF links from the regulatory website."""
    logger.info(f"Extracting PDF links from {url}...")
    
    try:
        logger.info(f"Fetching content from {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # Raise exception for non-200 status codes
        
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
            return []
        
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
            return []
        
        try:
            indices = [int(idx.strip()) for idx in indices_str.split(',') if idx.strip().isdigit()]
            
            # Get selected PDFs
            selected_pdfs = [pdf_links[idx] for idx in indices if idx < len(pdf_links)]
            logger.info(f"Selected {len(selected_pdfs)} PDFs")
            
            return selected_pdfs
        except Exception as e:
            logger.error(f"Error parsing LLM response for PDF selection: {str(e)}")
            # Return a subset of PDFs if parsing fails
            return pdf_links[:3] if pdf_links else []
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching website {url}: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Error extracting PDF links: {str(e)}")
        return []

def download_and_process_pdfs(pdf_urls):
    """Download PDFs and extract content."""
    logger.info("Downloading and processing PDFs...")
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
    
    return pdf_contents

def analyze_content(query, pdf_contents, client):
    """Analyze PDF content and generate answer with factual verification."""
    logger.info("Analyzing content...")
    
    # Check if we have any PDF contents to analyze
    if not pdf_contents:
        logger.warning("No PDF contents available for analysis")
        return "I'm sorry, but I couldn't find any relevant regulatory documents to answer your query. Please try a different query or select a specific market."
    
    # Combine all contents
    combined_text = ""
    for title, content in pdf_contents.items():
        combined_text += f"--- Document: {title} ---\n{content}\n\n"
    
    # Split text into chunks to stay within token limits
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=30000,  # Adjust based on token limit
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(combined_text)
    except ImportError:
        # Manual fallback text splitting if langchain is not available
        chunks = []
        max_chunk_size = 30000
        current_chunk = ""
        for line in combined_text.split("\n"):
            if len(current_chunk) + len(line) + 1 <= max_chunk_size:
                current_chunk += line + "\n"
            else:
                chunks.append(current_chunk)
                current_chunk = line + "\n"
        if current_chunk:
            chunks.append(current_chunk)
            
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
        return "I'm sorry, but I couldn't find any information in the regulatory documents that addresses your query. The documents I examined don't contain specific information about this topic."
    
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
        return final_answer
    except Exception as e:
        logger.error(f"Error generating final answer: {str(e)}")
        return "I apologize, but I encountered an error while processing your query. Please try again or rephrase your question."

def process_query(query, market=None, source=None, client=None):
    """Process a query using the simplified agent."""
    logger.info(f"Processing query: {query}, market: {market or 'Auto-detect'}, source: {source or 'Auto-detect'}")
    
    results = {
        "query": query,
        "market": market,
        "source": source,
        "selected_url": "",
        "pdf_urls": [],
        "pdf_contents": {},
        "final_answer": ""
    }
    
    # Step 1: Check if client is initialized
    if not client:
        logger.error("Groq client not initialized")
        results["final_answer"] = "I apologize, but I couldn't connect to the AI service. Please try again later."
        return results
    
    # Step 2: Determine market and source if not provided
    if not source:
        logger.info("Determining market and source...")
        determined_market, determined_source = get_market_and_source(query, client)
        
        # If market was provided but source wasn't, keep the provided market
        if market and not source:
            results["source"] = determined_source
        # If neither was provided, use both determined values
        elif not market and not source:
            results["market"] = determined_market
            results["source"] = determined_source
        
        if results["source"] == "NONE":
            logger.warning("Could not determine source automatically")
            results["final_answer"] = "I couldn't determine which regulatory source would be most relevant for your query. Please select a specific source and try again."
            return results
    
    # Step 3: Select URL based on the source
    if results["source"] in REGULATORY_WEBSITES:
        results["selected_url"] = REGULATORY_WEBSITES[results["source"]]
        logger.info(f"Selected URL: {results['selected_url']}")
    else:
        logger.warning(f"Source {results['source']} not found in regulatory websites")
        results["final_answer"] = f"I apologize, but I don't have information on the regulatory source '{results['source']}'. Please select one of the available sources."
        return results
    
    # Step 4: Extract PDF links
    results["pdf_urls"] = extract_pdf_links(results["selected_url"], query, client)
    if not results["pdf_urls"]:
        logger.warning("No relevant PDF links found")
        results["final_answer"] = "I couldn't find any relevant regulatory documents for your query. Please try a different query or select a different source."
        return results
    
    # Step 5: Download and process PDFs
    results["pdf_contents"] = download_and_process_pdfs(results["pdf_urls"])
    if not results["pdf_contents"]:
        logger.warning("No PDF contents could be extracted")
        results["final_answer"] = "I couldn't successfully download or extract content from the regulatory documents. Please try again later or with a different query."
        return results
    
    # Step 6: Analyze content and generate answer
    results["final_answer"] = analyze_content(query, results["pdf_contents"], client)
    
    return results

# Main application
def main():
    st.title("Automotive Regulations AI Agent")
    
    # Sidebar for logs
    st.sidebar.title("Execution Logs")
    log_placeholder = st.sidebar.empty()
    
    # Create a log handler that writes to the streamlit sidebar
    log_output = []
    
    class StreamlitLogHandler(logging.Handler):
        def emit(self, record):
            log_record = self.format(record)
            log_output.append(log_record)
            log_placeholder.text('\n'.join(log_output[-30:]))  # Keep only last 30 logs
    
    # Add the streamlit handler to the logger
    streamlit_handler = StreamlitLogHandler()
    logger.addHandler(streamlit_handler)
    
    # Initialize diagnostic logger
    diagnostic_logger = DiagnosticLogger()
    
    # Initialize Groq client
    client = initialize_groq_client()
    if not client:
        st.error("Failed to initialize Groq API client. Please check your API key configuration.")
        st.info("To configure the API key, you can use Streamlit secrets or environment variables.")
        return
    
    st.success("API key is configured. Ready to use!")
    
    # User query
    query = st.text_input("Enter your automotive regulatory query:")
    
    # Create market-to-sources mapping
    market_to_sources = create_market_to_sources_mapping()
    
    # Market and source selection
    st.subheader("Market and Source Selection")
    st.markdown("You can let the system automatically detect the market and relevant regulatory source, or select them manually.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sort markets with Global first and Other last
        sorted_markets = sorted(market_to_sources.keys())
        if "Global" in sorted_markets:
            sorted_markets.remove("Global")
            sorted_markets.insert(0, "Global")
        if "Other" in sorted_markets:
            sorted_markets.remove("Other")
            sorted_markets.append("Other")
        
        # Market selection dropdown
        market_options = ["Auto-detect"] + sorted_markets
        selected_market = st.selectbox("Select Market:", market_options)
    
    with col2:
        # Source selection dropdown, filtered by market
        if selected_market == "Auto-detect":
            source_options = ["Auto-detect"]
        else:
            source_options = ["Auto-detect"] + market_to_sources.get(selected_market, [])
        
        selected_source = st.selectbox("Select Regulatory Source:", source_options)
    
    if st.button("Process Query"):
        if query:
            with st.spinner("Processing your query..."):
                logger.info(f"Processing query: {query}")
                
                # Set market and source if manually selected
                market = None if selected_market == "Auto-detect" else selected_market
                source = None if selected_source == "Auto-detect" else selected_source
                
                # Get user IP for diagnostic logging (anonymized)
                user_ip = get_client_ip()
                user_id = diagnostic_logger.get_user_id(user_ip)
                
                # Process the query
                try:
                    result = process_query(query, market, source, client)
                    
                    # Log the session
                    accessed_documents = [title for title, _ in result.get("pdf_urls", [])]
                    diagnostic_logger.log_session(
                        user_id=user_id,
                        query=query,
                        market=result.get("market", "UNKNOWN"),
                        accessed_documents=accessed_documents,
                        answer=result.get("final_answer", "")
                    )
                    
                    # Display results
                    st.subheader("Results")
                    
                    # Display the market and source
                    if result.get("market") and result["market"] != "UNCLEAR":
                        st.write(f"Market: {result['market']}")
                    
                    if result.get("source") and result["source"] != "NONE":
                        st.write(f"Regulatory Source: {result['source']}")
                    else:
                        st.error("Could not determine relevant regulatory source automatically.")
                        
                        # Allow user to select a source if detection failed
                        if result.get("market") and result["market"] != "UNCLEAR":
                            source_options = market_to_sources.get(result["market"], [])
                            if source_options:
                                selected_source = st.selectbox("Please select a regulatory source:", source_options, key="source_select_after_error")
                                if st.button("Confirm Source", key="confirm_source_button"):
                                    result = process_query(query, result["market"], selected_source, client)
                                    
                                    # Log the session again with updated source
                                    accessed_documents = [title for title, _ in result.get("pdf_urls", [])]
                                    diagnostic_logger.log_session(
                                        user_id=user_id,
                                        query=query,
                                        market=result.get("market", "UNKNOWN"),
                                        accessed_documents=accessed_documents,
                                        answer=result.get("final_answer", "")
                                    )
                    
                    # Display URL
                    if result.get("selected_url"):
                        st.write(f"Website: {result['selected_url']}")
                    
                    # Display documents
                    st.subheader("Documents Analyzed")
                    if result.get("pdf_urls") and len(result["pdf_urls"]) > 0:
                        for title, url in result["pdf_urls"]:
                            st.write(f"- {title} ([link]({url}))")
                    else:
                        st.write("No relevant documents were found or selected.")
                    
                    # Display answer
                    st.subheader("Answer")
                    st.write(result.get("final_answer", "No answer was generated."))
                    
                except Exception as e:
                    error_message = str(e)
                    logger.error(f"Error during processing: {error_message}")
                    st.error(f"An error occurred while processing your query: {error_message}")
                    
                    # Log the error
                    diagnostic_logger.log_session(
                        user_id=user_id,
                        query=query,
                        market=market or "Auto-detect",
                        accessed_documents=[],
                        error=error_message
                    )
        else:
            st.warning("Please enter a query.")
    
    # The architecture diagram
    st.markdown("---")
    st.subheader("How This Application Works")
    
    # Create a collapsible section for the diagram
    with st.expander("Click to view the application architecture diagram"):
        try:
            # Generate polished diagram image
            with st.spinner("Generating process flow diagram..."):
                diagram_image = create_diagram_image()
                if diagram_image:
                    # Display the image
                    st.image(diagram_image, caption="Automotive Regulations AI Process Flow", use_column_width=True)
                    
                    # Add download option
                    img_str = get_image_base64(diagram_image)
                    if img_str:
                        href = f'<a href="data:image/png;base64,{img_str}" download="auto_regs_process_flow.png">Download Diagram</a>'
                        st.markdown(href, unsafe_allow_html=True)
                else:
                    raise Exception("Failed to generate diagram")
        except Exception as e:
            st.error(f"Could not generate diagram: {str(e)}")
            # Fall back to text-based diagram
            st.code("""
            User Input → Process Query → Initialize Agent → Processing Pipeline → Document Analysis → Generate Answer
                ↑                ↑                               ↑                      ↑                 ↑
            Market & Source Detection                      Groq LLM API connections (provides intelligence)
                                                                ↑                      ↑
                                                          Error Handling (monitors process)
                                                                                       ↓
                                                                                PDF Processing
            """)
    
    # Explanation of the diagram
    st.markdown("""
    ### Diagram Explanation
    This diagram shows how the Automotive Regulatory Document Assistant works:
    1. **User Interface**: You enter your query and the system automatically detects relevant market and source
    2. **Processing Pipeline**: The system analyzes your request and identifies appropriate regulatory documents
    3. **Document Analysis**: Relevant documents are found and processed to extract information
    4. **Answer Generation**: A comprehensive answer is created using only information from the documents
    """)
    
    # Add usage instructions
    st.markdown("---")
    st.markdown("""
    ## How to use this tool
    1. Enter your query about automotive regulations
    2. Either let the system automatically detect the relevant market and regulatory source, or select them manually
    3. Click "Process Query" to start the analysis
    4. The system will identify relevant documents from regulatory databases and provide an accurate answer based on their content only
    
    ## Example queries
    - "What are the crash test requirements for passenger vehicles in the US?"
    - "What are emission standards for electric vehicles in the EU from 2023 onwards?"
    - "Tell me about the latest safety regulations for autonomous vehicles in Japan"
    - "What are the approval procedures for importing used vehicles to Australia?"
    - "What kind of child restraint systems are required in China?"
    """)

# Run the application
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Critical error in application startup: {str(e)}")
        logger.exception("Critical application error")
