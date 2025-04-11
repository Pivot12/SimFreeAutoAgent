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
    
    # Extract explicit mentions of countries/regions first
    market_keywords = {
        "US": ["US", "USA", "United States", "America", "American", "NHTSA", "EPA", "DOT", "FMVSS", "Federal Motor Vehicle"],
        "EU": ["EU", "Europe", "European Union", "European", "EC", "ECE", "ACEA", "WVTA", "Euro"],
        "Global": ["Global", "International", "UNECE", "UN", "ISO", "IEC", "World", "Worldwide"],
        "UK": ["UK", "United Kingdom", "Britain", "British", "England", "DfT"],
        "China": ["China", "Chinese", "MIIT", "CCC", "GB standards"],
        "India": ["India", "Indian", "ARAI", "CMVR", "Bharat"],
        "Japan": ["Japan", "Japanese", "MLIT", "JASIC", "TRIAS"],
        "Canada": ["Canada", "Canadian", "CMVSS"],
        "Australia": ["Australia", "Australian", "ADR"],
        "Brazil": ["Brazil", "Brazilian", "INMETRO", "CONTRAN"],
        "South Korea": ["Korea", "Korean", "MOLIT", "KMVSS"],
        "Russia": ["Russia", "Russian", "Rosavtodor", "Customs Union", "EAC"],
        "Mexico": ["Mexico", "Mexican", "SCT", "NOM"],
        "South Africa": ["South Africa", "South African", "NRCS", "SABS"],
        "Argentina": ["Argentina", "Argentinian", "ANSV"]
    }
    
    # Direct match for market keywords in the query
    detected_market = None
    highest_match_count = 0
    
    for market, keywords in market_keywords.items():
        match_count = sum(1 for keyword in keywords if keyword.lower() in query.lower())
        # Also check for exact matches that might be a stronger signal
        exact_matches = sum(3 for keyword in keywords if f" {keyword.lower()} " in f" {query.lower()} ")
        
        total_score = match_count + exact_matches
        
        if total_score > highest_match_count:
            highest_match_count = total_score
            detected_market = market
    
    # If we have a strong direct match, use it directly - avoid LLM for simple cases
    if highest_match_count >= 2:
        logger.info(f"Direct keyword match detected market: {detected_market}")
        
        # Now determine source based on the detected market
        relevant_sources = []
        for source in REGULATORY_WEBSITES.keys():
            # Check if source name contains market name
            if detected_market.lower() in source.lower():
                relevant_sources.append(source)
                
        # For US fuel type queries, prioritize EPA and DOE
        if detected_market == "US" and any(fuel_term in query.lower() for fuel_term in ["fuel", "gas", "alternative", "gasoline", "diesel"]):
            for source in relevant_sources:
                if "EPA" in source or "Department of Energy" in source:
                    logger.info(f"Direct source match for US fuel query: {source}")
                    return detected_market, source
        
        # If we found relevant sources, use the first one
        if relevant_sources:
            logger.info(f"Using first relevant source for {detected_market}: {relevant_sources[0]}")
            return detected_market, relevant_sources[0]
        
        # If no sources found for market, just return the market and let the fallback mechanism handle it
        return detected_market, "NONE"
    
    # If direct matching failed or wasn't strong enough, use the LLM
    prompt = f"""
    Based on the following query about automotive regulations, determine:
    1. Which market (country/region) the user is interested in
    2. Which regulatory source would be most relevant to answer their query

    User query: {query}
    
    For US fuel type regulations, be sure to consider the US Environmental Protection Agency (EPA) and Department of Energy as they regulate vehicle fuels.
    
    Respond in the following format:
    MARKET: [market name or "UNCLEAR"]
    SOURCE: [exact name of the most relevant regulatory source or "NONE" if unclear]
    
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
                if source.lower() in key.lower() or key.lower() in source.lower():
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

def clean_url(url):
    """
    Clean and validate a URL, ensuring it has the proper scheme and no invisible characters.
    """
    # Remove invisible characters
    url = url.replace('\u200b', '').strip()
    
    # Add https:// if missing
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    return url

def get_fallback_source(market):
    """
    Get a fallback source for a given market if the primary source fails.
    
    This is a backup mechanism to ensure the agent can continue processing
    even if the primary regulatory source website is unreachable.
    """
    if not market or market == "UNCLEAR":
        return None
    
    # Map of markets to alternative sources
    fallback_sources = {
        "US": ["🇺🇸 United States - National Highway Traffic Safety Administration (NHTSA)", 
               "US Environmental Protection Agency (EPA) – Vehicle Regulations"],
        "EU": ["EU European Commission", 
               "European Automobile Manufacturers' Association (ACEA)", 
               "European Free Trade Association (EFTA) – Vehicle Regulations"],
        "Global": ["Global & Regional Authorities UNECE",
                  "International Organization for Standardization (ISO) – Road Vehicles"],
        "China": ["🇨🇳 China Ministry of Industry and Information Technology (MIIT)"],
        "India": ["🇮🇳 India Automotive Research Association of India (ARAI)", 
                 "Central Motor Vehicle Rules (CMVR)"],
        "Japan": ["🇯🇵 Japan Ministry of Land, Infrastructure, Transport and Tourism (MLIT)"]
    }
    
    # Get fallback sources for the market
    sources = fallback_sources.get(market, [])
    
    # Filter to only include sources that exist in our regulatory websites
    valid_sources = [s for s in sources if s in REGULATORY_WEBSITES]
    
    return valid_sources[0] if valid_sources else None

def extract_pdf_links(url, query, client):
    """Extract PDF links from the regulatory website by properly traversing the site structure."""
    logger.info(f"Extracting real PDF links from {url}...")
    
    try:
        # Clean and validate the URL before using it
        cleaned_url = clean_url(url)
        
        if cleaned_url != url:
            logger.info(f"URL cleaned: {url} -> {cleaned_url}")
            url = cleaned_url
        
        logger.info(f"Fetching content from {url}")
        
        # Add comprehensive request headers to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Referer': 'https://www.google.com/',  # Pretend we came from Google
            'Cache-Control': 'max-age=0',
            'TE': 'Trailers',
            'DNT': '1'
        }
        
        session = requests.Session()
        
        # Try common URL variants if the original fails
        urls_to_try = [
            url,
            url.replace("https://www.", "https://"),
            "https://www." + url.replace("https://", "") if url.startswith("https://") else url,
            # UNECE specific handling
            url.replace("unece.org/trans/main/wp29/wp29regs.html", "unece.org/transport/vehicle-regulations-wp29") if "unece" in url else url,
            "https://unece.org/transport/vehicle-regulations-wp29" if "unece" in url else url
        ]
        
        response = None
        successful_url = None
        
        for try_url in urls_to_try:
            try:
                logger.info(f"Attempting to access: {try_url}")
                response = session.get(try_url, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    logger.info(f"Successfully accessed: {try_url}")
                    successful_url = try_url
                    break
                    
                logger.warning(f"Failed to access {try_url}: {response.status_code}")
            except requests.exceptions.RequestException as e:
                logger.error(f"Error accessing {try_url}: {str(e)}")
                continue
                
        if not response or response.status_code != 200:
            logger.error(f"Failed to access any URL variant for {url}")
            return []
        
        # Update the URL to the successful one
        url = successful_url
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all links on the page
        links = soup.find_all('a')
        
        # Extract direct PDF links first
        pdf_links = []
        for link in links:
            href = link.get('href')
            if not href:
                continue
                
            # Check if this is a PDF or a publications page
            is_pdf = href.lower().endswith('.pdf')
            is_publications_page = any(keyword in href.lower() for keyword in 
                                        ['publication', 'document', 'regulation', 'standard', 
                                         'directive', 'legislation', 'report', 'guideline'])
            
            if is_pdf or is_publications_page:
                # Make sure we have absolute URLs
                full_url = href
                if not href.startswith(('http://', 'https://')):
                    # Handle relative URLs properly
                    if href.startswith('/'):
                        # Get base domain
                        from urllib.parse import urlparse
                        parsed_url = urlparse(url)
                        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
                        full_url = base_url + href
                    else:
                        # Relative to current path
                        if url.endswith('/'):
                            full_url = url + href
                        else:
                            last_slash = url.rfind('/')
                            if '.' in url[last_slash:]:  # URL points to a file
                                base_url = url[:last_slash+1]
                            else:  # URL points to a directory
                                base_url = url + ('/' if not url.endswith('/') else '')
                            full_url = base_url + href
                
                # Clean the URL
                full_url = clean_url(full_url)
                
                # Add PDFs directly to our list
                if is_pdf:
                    title = link.text.strip() if link.text.strip() else os.path.basename(href)
                    pdf_links.append((title, full_url))
                    logger.info(f"Found PDF link: {title} - {full_url}")
                    
                # If it's a publications page, we'll check it for more PDFs
                elif is_publications_page and full_url != url:  # Avoid checking the same page
                    logger.info(f"Found publications page: {full_url}")
                    try:
                        # Don't check pages we've already visited to avoid loops
                        pub_response = session.get(full_url, headers=headers, timeout=30)
                        
                        if pub_response.status_code == 200:
                            pub_soup = BeautifulSoup(pub_response.text, 'html.parser')
                            pub_links = pub_soup.find_all('a')
                            
                            for pub_link in pub_links:
                                pub_href = pub_link.get('href')
                                if pub_href and pub_href.lower().endswith('.pdf'):
                                    # Process similarly to above
                                    if pub_href.startswith(('http://', 'https://')):
                                        pub_full_url = pub_href
                                    elif pub_href.startswith('/'):
                                        from urllib.parse import urlparse
                                        parsed_url = urlparse(full_url)
                                        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
                                        pub_full_url = base_url + pub_href
                                    else:
                                        if full_url.endswith('/'):
                                            pub_full_url = full_url + pub_href
                                        else:
                                            last_slash = full_url.rfind('/')
                                            if '.' in full_url[last_slash:]:  # URL points to a file
                                                base_url = full_url[:last_slash+1]
                                            else:  # URL points to a directory
                                                base_url = full_url + ('/' if not full_url.endswith('/') else '')
                                            pub_full_url = base_url + pub_href
                                    
                                    pub_full_url = clean_url(pub_full_url)
                                    
                                    pub_title = pub_link.text.strip() if pub_link.text.strip() else os.path.basename(pub_href)
                                    pdf_links.append((pub_title, pub_full_url))
                                    logger.info(f"Found PDF link on publications page: {pub_title} - {pub_full_url}")
                        else:
                            logger.warning(f"Failed to access publications page {full_url}: {pub_response.status_code}")
                    except Exception as pub_error:
                        logger.error(f"Error processing publications page {full_url}: {str(pub_error)}")
        
        logger.info(f"Found {len(pdf_links)} PDF links")
        
        if not pdf_links:
            logger.warning("No PDF links found on the regulatory website")
            
            # If no direct PDF links found, try to find links to publications or document sections
            doc_section_links = []
            for link in links:
                href = link.get('href')
                text = link.text.lower() if link.text else ""
                
                if not href:
                    continue
                    
                # Keywords that might indicate document sections
                doc_section_indicators = [
                    'publication', 'document', 'library', 'resource', 'download',
                    'regulation', 'directive', 'legislation', 'report', 'standard', 
                    'guideline', 'technical', 'official', 'legal', 'policy'
                ]
                
                if any(indicator in text or indicator in href.lower() for indicator in doc_section_indicators):
                    # Process URL the same way as above
                    if href.startswith(('http://', 'https://')):
                        full_url = href
                    elif href.startswith('/'):
                        from urllib.parse import urlparse
                        parsed_url = urlparse(url)
                        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
                        full_url = base_url + href
                    else:
                        if url.endswith('/'):
                            full_url = url + href
                        else:
                            last_slash = url.rfind('/')
                            if '.' in url[last_slash:]:  # URL points to a file
                                base_url = url[:last_slash+1]
                            else:  # URL points to a directory
                                base_url = url + ('/' if not url.endswith('/') else '')
                            full_url = base_url + href
                    
                    full_url = clean_url(full_url)
                    
                    # Avoid adding the same URL twice
                    if full_url not in [link for _, link in doc_section_links]:
                        doc_section_links.append((text, full_url))
            
            # Check document sections for PDFs
            for section_text, section_url in doc_section_links[:5]:  # Limit to first 5 to avoid too many requests
                if section_url == url:  # Skip the current URL to avoid loops
                    continue
                    
                logger.info(f"Checking document section: {section_text} at {section_url}")
                
                try:
                    section_response = session.get(section_url, headers=headers, timeout=30)
                    
                    if section_response.status_code == 200:
                        section_soup = BeautifulSoup(section_response.text, 'html.parser')
                        section_links = section_soup.find_all('a')
                        
                        for link in section_links:
                            href = link.get('href')
                            if href and href.lower().endswith('.pdf'):
                                # Process URL the same way as above
                                if href.startswith(('http://', 'https://')):
                                    full_url = href
                                elif href.startswith('/'):
                                    from urllib.parse import urlparse
                                    parsed_url = urlparse(section_url)
                                    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
                                    full_url = base_url + href
                                else:
                                    if section_url.endswith('/'):
                                        full_url = section_url + href
                                    else:
                                        last_slash = section_url.rfind('/')
                                        if '.' in section_url[last_slash:]:  # URL points to a file
                                            base_url = section_url[:last_slash+1]
                                        else:  # URL points to a directory
                                            base_url = section_url + ('/' if not section_url.endswith('/') else '')
                                        full_url = base_url + href
                                
                                full_url = clean_url(full_url)
                                
                                title = link.text.strip() if link.text.strip() else os.path.basename(href)
                                pdf_links.append((title, full_url))
                                logger.info(f"Found PDF link on document section page: {title} - {full_url}")
                    else:
                        logger.warning(f"Failed to access document section {section_url}: {section_response.status_code}")
                except Exception as section_error:
                    logger.error(f"Error checking document section {section_url}: {str(section_error)}")
        
        logger.info(f"Total PDF links found: {len(pdf_links)}")
        
        if not pdf_links:
            logger.warning("No PDF links found after checking all potential document sections")
            return []
        
        # Use LLM to select relevant PDFs based on the query (only if we have too many PDFs)
        if len(pdf_links) > 5:
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
                # Return a few PDFs anyway since we found them on the site
                return pdf_links[:3]
            
            try:
                indices = [int(idx.strip()) for idx in indices_str.split(',') if idx.strip().isdigit()]
                
                # Get selected PDFs
                selected_pdfs = [pdf_links[idx] for idx in indices if idx < len(pdf_links)]
                logger.info(f"Selected {len(selected_pdfs)} PDFs based on relevance")
                
                return selected_pdfs if selected_pdfs else pdf_links[:3]
            except Exception as e:
                logger.error(f"Error parsing LLM response for PDF selection: {str(e)}")
                # Return a subset of PDFs if parsing fails
                return pdf_links[:3]
        else:
            # If we have a reasonable number of PDFs, just return all of them
            return pdf_links
    
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
            
            # Skip example URLs completely - NO synthetic content generation
            if "example.org" in url or "example.com" in url:
                logger.warning(f"Skipping example URL: {url}")
                continue
                
            # Clean the URL before using it
            url = clean_url(url)
            
            # Try to download the PDF with timeout and retries
            max_retries = 3
            retry_count = 0
            response = None
            
            while retry_count < max_retries:
                try:
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    break
                except requests.exceptions.RequestException as e:
                    retry_count += 1
                    logger.warning(f"Retry {retry_count}/{max_retries} - Error downloading PDF: {str(e)}")
                    
                    # Try alternative URL formats if original failed
                    if retry_count == 1 and url.startswith("https://www."):
                        url = url.replace("https://www.", "https://")
                    elif retry_count == 2 and url.startswith("https://"):
                        url = "https://www." + url[8:]
                    
                    if retry_count >= max_retries:
                        logger.error(f"Failed to download PDF after {max_retries} retries: {url}")
                        break
                    
                    # Wait before retrying
                    time.sleep(1)
            
            # If we couldn't get a response, skip this PDF
            if not response or response.status_code != 200:
                logger.warning(f"Failed to download PDF: {url}, status code: {response.status_code if response else 'No response'}")
                continue
                
            # Check content type to ensure it's a PDF
            content_type = response.headers.get('Content-Type', '').lower()
            
            # If content type header isn't application/pdf but URL ends with .pdf, try anyway
            if 'application/pdf' not in content_type and not url.lower().endswith('.pdf'):
                logger.warning(f"URL does not return PDF content: {url}, content type: {content_type}")
                
                # Check if it's HTML content - if so, we might be able to extract text
                if 'text/html' in content_type:
                    logger.info(f"URL returned HTML content, attempting to extract text")
                    try:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Try to extract main content
                        main_content = ""
                        
                        # Look for common content containers
                        content_elements = soup.select("main, article, .content, #content, .main-content")
                        if content_elements:
                            for element in content_elements:
                                main_content += element.get_text() + "\n\n"
                        else:
                            # If no common containers found, just get the body text
                            body = soup.find('body')
                            if body:
                                main_content = body.get_text()
                        
                        # Clean up the text
                        main_content = re.sub(r'\s+', ' ', main_content).strip()
                        
                        if main_content:
                            pdf_contents[f"{title} (HTML content)"] = main_content
                            successful_downloads += 1
                            logger.info(f"Successfully extracted HTML content: {title}, extracted {len(main_content)} characters")
                    except Exception as e:
                        logger.error(f"Error extracting HTML content: {str(e)}")
                
                # Continue to next URL since this isn't a PDF
                continue
            
            # Now process the PDF content
            pdf_file = io.BytesIO(response.content)
            
            # Read PDF content
            try:
                reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                
                # Get total number of pages
                total_pages = len(reader.pages)
                logger.info(f"PDF has {total_pages} pages")
                
                # Process all pages or a subset for very large documents
                max_pages = min(100, total_pages)  # Process up to 100 pages
                
                for i in range(max_pages):
                    try:
                        page = reader.pages[i]
                        page_text = page.extract_text()
                        if page_text:  # Only add if text was successfully extracted
                            text += page_text + "\n\n"
                    except Exception as page_error:
                        logger.error(f"Error extracting text from page {i}: {str(page_error)}")
                
                # Add a note if we didn't process all pages
                if total_pages > max_pages:
                    text += f"\n\n[Note: Only the first {max_pages} pages of {total_pages} total pages were processed.]"
                
                # Only add if we got actual content
                if text.strip():
                    pdf_contents[title] = text
                    successful_downloads += 1
                    logger.info(f"Successfully processed PDF: {title}, extracted {len(text)} characters")
                else:
                    logger.warning(f"No text could be extracted from PDF: {title}")
                    
                    # If no text could be extracted, the PDF might be scanned/image-based
                    # In a production system, you would use OCR here
                    logger.info(f"PDF may be image-based, OCR would be needed: {title}")
            except Exception as e:
                logger.error(f"Error reading PDF {title}: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error processing PDF {title}: {str(e)}")
    
    if successful_downloads == 0:
        logger.warning("No PDFs were successfully downloaded and processed")
    
    return pdf_contents

import time
import importlib.util

# Add token counting functionality - check if tiktoken is available
def get_token_count(text, model="llama-3.3-70b-versatile"):
    """Estimate token count for text using tiktoken if available, or a simple approximation."""
    if importlib.util.find_spec("tiktoken"):
        import tiktoken
        try:
            encoder = tiktoken.encoding_for_model(model)
            return len(encoder.encode(text))
        except Exception:
            # Fallback to approximation if tiktoken fails
            pass
    
    # Simple approximation: 1 token ≈ 4 characters
    return len(text) // 4

def analyze_content_with_token_management(query, pdf_contents, client):
    """Analyze PDF content and generate answer with token limit management."""
    logger.info("Analyzing content with token management...")
    
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
            chunk_size=5000,  # Smaller chunks to stay well under token limit
            chunk_overlap=200,
            length_function=get_token_count,
        )
        chunks = text_splitter.split_text(combined_text)
    except ImportError:
        # Manual fallback text splitting if langchain is not available
        chunks = []
        max_token_size = 5000  # Maximum tokens per chunk
        current_chunk = ""
        for line in combined_text.split("\n"):
            line_tokens = get_token_count(line)
            chunk_tokens = get_token_count(current_chunk)
            
            if chunk_tokens + line_tokens <= max_token_size:
                current_chunk += line + "\n"
            else:
                chunks.append(current_chunk)
                current_chunk = line + "\n"
        if current_chunk:
            chunks.append(current_chunk)
            
    logger.info(f"Split content into {len(chunks)} chunks")
    
    # Process chunks and collect insights with citations
    insights = []
    
    # Token management variables
    tokens_per_minute_limit = 6000
    tokens_used_in_minute = 0
    minute_start_time = time.time()
    
    for i, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {i+1}/{len(chunks)}")
        
        # Estimate tokens for this request
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
        
        # Calculate tokens for this prompt
        estimated_prompt_tokens = get_token_count(prompt)
        estimated_response_tokens = 1000  # Estimate for response
        estimated_total_tokens = estimated_prompt_tokens + estimated_response_tokens
        
        # Check if we need to reset the minute counter
        current_time = time.time()
        if current_time - minute_start_time >= 60:
            tokens_used_in_minute = 0
            minute_start_time = current_time
        
        # Check if adding this request would exceed our limit
        if tokens_used_in_minute + estimated_total_tokens > tokens_per_minute_limit:
            # Calculate wait time needed to stay under limit
            wait_seconds = 60 - (current_time - minute_start_time)
            if wait_seconds > 0:
                logger.info(f"Approaching token limit, waiting {wait_seconds:.1f} seconds before continuing")
                time.sleep(wait_seconds)
                tokens_used_in_minute = 0
                minute_start_time = time.time()
        
        try:
            logger.info(f"Calling LLM for chunk {i+1} analysis...")
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )
            chunk_insights = response.choices[0].message.content
            
            # Update token usage
            tokens_used = estimated_prompt_tokens + get_token_count(chunk_insights)
            tokens_used_in_minute += tokens_used
            logger.info(f"Used approximately {tokens_used} tokens for chunk {i+1}")
            
            # Only add if there are actually insights found
            if "NO RELEVANT INFORMATION FOUND IN THIS CHUNK" not in chunk_insights:
                insights.append(chunk_insights)
            logger.info(f"Successfully analyzed chunk {i+1}")
        except Exception as e:
            logger.error(f"Error processing chunk {i+1}: {str(e)}")
            # If we hit a rate limit, wait and retry
            if "rate_limit_exceeded" in str(e) or "Request too large" in str(e):
                logger.info("Hit rate limit, waiting 60 seconds before retrying")
                time.sleep(60)
                tokens_used_in_minute = 0
                minute_start_time = time.time()
                
                try:
                    # Retry the request with a shorter chunk if possible
                    shortened_chunk = chunk[:len(chunk)//2] + "..."
                    shortened_prompt = prompt.replace(chunk, shortened_chunk)
                    
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": shortened_prompt}],
                        max_tokens=500  # Reduced token limit for retry
                    )
                    chunk_insights = response.choices[0].message.content
                    
                    # Only add if there are actually insights found
                    if "NO RELEVANT INFORMATION FOUND IN THIS CHUNK" not in chunk_insights:
                        insights.append(chunk_insights)
                    logger.info(f"Successfully analyzed chunk {i+1} (shortened) on retry")
                except Exception as retry_error:
                    logger.error(f"Error on retry for chunk {i+1}: {str(retry_error)}")
    
    # If no insights found at all, return a "no answer" response
    if not insights:
        logger.warning("No relevant insights found in any document chunks")
        return "I'm sorry, but I couldn't find any information in the regulatory documents that addresses your query. The documents I examined don't contain specific information about this topic."
    
    # Combine insights and generate final answer with factual verification
    combined_insights = "\n\n".join(insights)
    
    # Reset token counter before final answer generation
    current_time = time.time()
    if current_time - minute_start_time >= 60:
        tokens_used_in_minute = 0
        minute_start_time = current_time
    
    # Wait if needed to avoid rate limits
    if tokens_used_in_minute > tokens_per_minute_limit * 0.7:  # If we're at 70% of the limit, wait
        wait_seconds = 60 - (current_time - minute_start_time)
        if wait_seconds > 0:
            logger.info(f"Waiting {wait_seconds:.1f} seconds before generating final answer")
            time.sleep(wait_seconds)
        tokens_used_in_minute = 0
        minute_start_time = time.time()
    
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
    
    # Check if final prompt would exceed token limit
    final_prompt_tokens = get_token_count(prompt)
    if final_prompt_tokens > tokens_per_minute_limit - 2000:  # Leaving room for response
        # Truncate combined insights to fit within limits
        max_insights_tokens = tokens_per_minute_limit - 4000  # Reserve tokens for prompt template and response
        truncated_insights = ""
        total_tokens = 0
        
        for insight in insights:
            insight_tokens = get_token_count(insight)
            if total_tokens + insight_tokens <= max_insights_tokens:
                truncated_insights += insight + "\n\n"
                total_tokens += insight_tokens
            else:
                break
        
        # Recreate prompt with truncated insights
        prompt = prompt.replace(combined_insights, truncated_insights + "\n[Note: Some insights were truncated due to length constraints]")
    
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
        
        # If we hit token limits, try with a smaller prompt
        if "rate_limit_exceeded" in str(e) or "Request too large" in str(e):
            logger.info("Hit rate limit with final answer, trying with shortened insights")
            time.sleep(60)  # Wait a minute to reset rate limits
            
            # Create a shorter version with just 1-2 most relevant insights
            shortened_insights = ""
            if insights:
                shortened_insights = insights[0]
                if len(insights) > 1:
                    shortened_insights += "\n\n[Additional insights omitted due to length constraints]"
            
            shortened_prompt = f"""
            Based on the following insights extracted from automotive regulatory documents, provide a concise answer to the user's query.
            
            User query: {query}
            
            Key insight from documents:
            {shortened_insights}
            
            Important instructions:
            1. Your answer MUST be derived ONLY from the document insights provided above
            2. Include citations to the specific document source for each claim
            3. Keep your answer brief and focused on the most relevant information
            4. Do not make claims not supported by the document evidence
            5. Format citations as: [Source Document Name]
            
            Provide a factual answer focusing ONLY on what's present in the automotive regulations documents.
            """
            
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": shortened_prompt}],
                    max_tokens=1000
                )
                final_answer = response.choices[0].message.content
                logger.info("Generated shortened final answer successfully")
                return final_answer
            except Exception as retry_error:
                logger.error(f"Error generating shortened final answer: {str(retry_error)}")
                return "I apologize, but I encountered an error while processing your query. Please try again with a more specific question or select a different regulatory source."
        
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
            
            # Try to find a source based on the market if we have one
            if results["market"] and results["market"] != "UNCLEAR":
                fallback_source = get_fallback_source(results["market"])
                if fallback_source:
                    logger.info(f"Using fallback source for market {results['market']}: {fallback_source}")
                    results["source"] = fallback_source
                else:
                    results["final_answer"] = f"I couldn't determine which regulatory source would be most relevant for your query about {results['market']}. Please select a specific source and try again."
                    return results
            else:
                results["final_answer"] = "I couldn't determine which regulatory source would be most relevant for your query. Please select a specific source and try again."
                return results
    
    # Step 3: Select URL based on the source
    if results["source"] in REGULATORY_WEBSITES:
        results["selected_url"] = REGULATORY_WEBSITES[results["source"]]
        logger.info(f"Selected URL: {results['selected_url']}")
    else:
        logger.warning(f"Source {results['source']} not found in regulatory websites")
        
        # Try to find a close match for the source
        close_match = None
        for key in REGULATORY_WEBSITES.keys():
            if results["source"].lower() in key.lower() or key.lower() in results["source"].lower():
                close_match = key
                break
        
        if close_match:
            logger.info(f"Found close match for source: {close_match}")
            results["source"] = close_match
            results["selected_url"] = REGULATORY_WEBSITES[close_match]
        else:
            # Try to find any source for the market if we have one
            if results["market"] and results["market"] != "UNCLEAR":
                fallback_source = get_fallback_source(results["market"])
                if fallback_source:
                    logger.info(f"Using fallback source for market {results['market']}: {fallback_source}")
                    results["source"] = fallback_source
                    results["selected_url"] = REGULATORY_WEBSITES[fallback_source]
                else:
                    results["final_answer"] = f"I apologize, but I don't have information on the regulatory source '{results['source']}'. Please select one of the available sources."
                    return results
            else:
                results["final_answer"] = f"I apologize, but I don't have information on the regulatory source '{results['source']}'. Please select one of the available sources."
                return results
    
    # Step 4: Extract PDF links
    max_attempts = 3
    attempt = 1
    while attempt <= max_attempts:
        logger.info(f"Attempt {attempt}/{max_attempts} to extract PDF links from {results['selected_url']}")
        results["pdf_urls"] = extract_pdf_links(results["selected_url"], query, client)
        
        if results["pdf_urls"]:
            break
        
        # If we couldn't find any PDFs, try an alternative source
        if attempt < max_attempts:
            if results["market"] and results["market"] != "UNCLEAR":
                # Try to find sources for this market
                market_sources = []
                for source_name in REGULATORY_WEBSITES.keys():
                    if results["market"].lower() in source_name.lower():
                        if source_name != results["source"]:  # Don't use the same source again
                            market_sources.append(source_name)
                
                if market_sources:
                    # Use the next available source
                    next_source = market_sources[min(attempt-1, len(market_sources)-1)]
                    logger.info(f"Trying alternative source: {next_source}")
                    results["source"] = next_source
                    results["selected_url"] = REGULATORY_WEBSITES[next_source]
                else:
                    # No more sources to try for this market
                    break
            else:
                # No market information to try alternative sources
                break
        
        attempt += 1
    
    if not results["pdf_urls"]:
        logger.warning("No relevant PDF links found after all attempts")
        results["final_answer"] = "I couldn't find any relevant regulatory documents for your query. Please try a different query with more specific terms related to automotive regulations."
        return results
    
    # Step 5: Download and process PDFs
    results["pdf_contents"] = download_and_process_pdfs(results["pdf_urls"])
    if not results["pdf_contents"]:
        logger.warning("No PDF contents could be extracted")
        
        # Try using a different approach: get information from the website itself
        logger.info("Attempting to extract information directly from the website")
        
        try:
            # Clean and validate the URL before using it
            url = clean_url(results["selected_url"])
            
            # Fetch content
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract text from the website
            text_content = soup.get_text()
            
            # Clean up the text (remove excess whitespace, etc.)
            text_content = re.sub(r'\s+', ' ', text_content).strip()
            
            # Create a content from the website text
            if text_content:
                results["pdf_contents"] = {
                    f"Website content from {results['source']}": text_content
                }
                logger.info(f"Successfully extracted text directly from website: {len(text_content)} characters")
        except Exception as e:
            logger.error(f"Error extracting website content: {str(e)}")
        
        # If we still don't have content, return an appropriate message
        if not results["pdf_contents"]:
            results["final_answer"] = "I couldn't successfully download or extract content from the regulatory documents. Please try again later or with a different query."
            return results
    
    # Step 6: Analyze content and generate answer with token limit management
    results["final_answer"] = analyze_content_with_token_management(query, results["pdf_contents"], client)
    
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
