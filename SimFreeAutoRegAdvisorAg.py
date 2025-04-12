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
def enhanced_market_detection(query, client):
    """
    Enhanced market detection using NLP techniques and a multi-stage approach.
    
    This function uses a combination of rule-based pattern matching, keyword analysis,
    and LLM-based classification to determine the regulatory market with high accuracy.
    """
    logger.info("Starting enhanced market detection for query...")
    
    # Stage 1: Direct keyword matching with weighted scoring
    market_patterns = {
        "US": {
            "aliases": ["US", "USA", "United States", "America", "American", "U.S.", "U.S.A."],
            "agencies": ["NHTSA", "EPA", "DOT", "FMVSS", "Federal Motor Vehicle", "DOE", "CAFE"],
            "regulations": ["CFR", "Title 49", "Part 571", "Federal Register"]
        },
        "EU": {
            "aliases": ["EU", "Europe", "European Union", "European", "E.U."],
            "agencies": ["EC", "ECE", "ACEA", "European Commission"],
            "regulations": ["WVTA", "Euro NCAP", "Type Approval", "Euro", "Directive", "EC Regulation"]
        },
        "Global": {
            "aliases": ["Global", "International", "Worldwide", "World"],
            "agencies": ["UNECE", "UN", "United Nations", "ISO", "IEC", "WHO"],
            "regulations": ["GTR", "Global Technical Regulation", "UN Regulation"]
        },
        "UK": {
            "aliases": ["UK", "United Kingdom", "Britain", "British", "England", "U.K."],
            "agencies": ["DfT", "DVSA", "VCA"],
            "regulations": ["Type Approval", "SVA", "British Standard"]
        },
        "China": {
            "aliases": ["China", "Chinese", "PRC"],
            "agencies": ["MIIT", "CCC", "CATARC"],
            "regulations": ["GB standard", "GB/T", "GB standards", "Chinese standard"]
        },
        "India": {
            "aliases": ["India", "Indian"],
            "agencies": ["ARAI", "CMVR", "BIS"],
            "regulations": ["AIS", "Bharat", "IS"]
        },
        "Japan": {
            "aliases": ["Japan", "Japanese"],
            "agencies": ["MLIT", "JASIC", "JAMA"],
            "regulations": ["TRIAS", "J-NCAP", "Japanese standard"]
        },
        "Canada": {
            "aliases": ["Canada", "Canadian"],
            "agencies": ["Transport Canada", "TC"],
            "regulations": ["CMVSS", "Canadian Motor Vehicle"]
        },
        "Australia": {
            "aliases": ["Australia", "Australian", "Aus", "AU"],
            "agencies": ["ADR", "ANCAP"],
            "regulations": ["Australian Design Rules", "Vehicle Standards"]
        }
    }
    
    # Initialize scores for each market
    market_scores = {market: 0 for market in market_patterns.keys()}
    
    # Create a normalized version of the query for matching
    query_normalized = ' ' + query.lower() + ' '
    
    # Calculate scores based on keyword matches
    for market, patterns in market_patterns.items():
        # Check for country/region name matches (high weight)
        for alias in patterns["aliases"]:
            # Check for exact words with word boundaries
            pattern = r'\b' + re.escape(alias.lower()) + r'\b'
            matches = re.findall(pattern, query_normalized)
            if matches:
                # Exact market name is a strong signal
                market_scores[market] += len(matches) * 10
                logger.info(f"Found market alias match for {market}: {alias}")
        
        # Check for regulatory agency mentions (medium-high weight)
        for agency in patterns["agencies"]:
            if agency.lower() in query_normalized:
                market_scores[market] += 8
                logger.info(f"Found agency match for {market}: {agency}")
        
        # Check for regulation mentions (medium weight)
        for regulation in patterns["regulations"]:
            if regulation.lower() in query_normalized:
                market_scores[market] += 5
                logger.info(f"Found regulation match for {market}: {regulation}")
    
    # Stage 2: Check for specific topic-market associations
    topic_market_mapping = {
        "fuel economy": "US",
        "cafe standard": "US",
        "crash test": "US",
        "emission standard": "US",
        "alternative fuel": "US",
        "zero emission": "US",
        "type approval": "EU",
        "exhaust emission": "EU",
        "euro ncap": "EU",
        "gb standard": "China",
        "ais standard": "India",
        "bharat stage": "India",
        "cmvss": "Canada",
        "adr": "Australia"
    }
    
    for topic, market in topic_market_mapping.items():
        if topic in query_normalized:
            market_scores[market] += 7
            logger.info(f"Found topic-market association: {topic} -> {market}")
    
    # Stage 3: Handle specific cases 
    # US fuel types is a common question that needs special handling
    if any(term in query_normalized for term in [" fuel ", "gasoline", "diesel", "electric vehicle", "ev ", "hydrogen"]):
        if " in us" in query_normalized or "united states" in query_normalized:
            market_scores["US"] += 15
            logger.info("Detected US fuel type question with explicit US mention")
        else:
            # If 'fuel' is mentioned without a specific market, slightly favor US
            market_scores["US"] += 5
            logger.info("Detected fuel type question, adding weight to US market")
    
    # Stage 4: Determine if we have a clear winner
    max_score = max(market_scores.values())
    top_markets = [market for market, score in market_scores.items() if score == max_score]
    
    logger.info(f"Market scores: {market_scores}")
    logger.info(f"Top markets: {top_markets}")
    
    # If we have a single clear winner with score above threshold
    if len(top_markets) == 1 and max_score >= 10:
        detected_market = top_markets[0]
        logger.info(f"Clear market detected through pattern matching: {detected_market}")
        return detected_market
    
    # If we have a tie or no strong signal, use the LLM for more sophisticated analysis
    logger.info("No clear market detected through pattern matching, using LLM...")
    
    # Create a more effective LLM prompt with guidance and examples
    llm_prompt = f"""
    Based on the following query about automotive regulations, determine which country or region's regulatory framework is most relevant.

    Query: "{query}"
    
    Consider these examples:
    - "What are the FMVSS requirements for passenger vehicles?" → US
    - "What Euro 6 emission standards apply to diesel engines?" → EU
    - "What are the GB standards for electric vehicles in 2023?" → China
    - "What regulations govern autonomous vehicles in Japan?" → Japan
    
    The query may mention specific regulatory bodies (e.g., NHTSA, EPA for US; EC for EU), 
    standards (e.g., FMVSS for US; Euro NCAP for EU), or regulations (e.g., Title 49 CFR for US).
    
    If the query doesn't clearly indicate a market, consider which market would be most relevant 
    based on the subject matter. For questions about fuel types, CAFE standards, or emissions 
    without a specified market, the US regulations are often most relevant.
    
    Return only the market name from this list: US, EU, Global, UK, China, India, Japan, Canada, Australia
    
    If truly unable to determine, return "UNCLEAR".
    """
    
    try:
        # Call LLM with improved prompt
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": llm_prompt}],
            max_tokens=10
        )
        
        llm_market = response.choices[0].message.content.strip()
        logger.info(f"LLM detected market: {llm_market}")
        
        # If LLM returns a valid market, use it
        valid_markets = list(market_patterns.keys())
        if llm_market in valid_markets:
            return llm_market
        
        # If LLM says UNCLEAR but we have some signals from keyword matching, use the highest scoring market
        if llm_market == "UNCLEAR" and max_score > 0:
            detected_market = top_markets[0]
            logger.info(f"Using highest scoring market from pattern matching: {detected_market}")
            return detected_market
            
        # If all else fails
        return "UNCLEAR"
    except Exception as e:
        logger.error(f"Error in LLM market detection: {str(e)}")
        
        # Fall back to highest scoring market if available
        if max_score > 0:
            detected_market = top_markets[0]
            logger.info(f"Falling back to highest scoring market due to LLM error: {detected_market}")
            return detected_market
        
        return "UNCLEAR"

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

def process_multi_format_content(url, query, client):
    """Process documents in multiple formats, not just PDFs."""
    logger.info(f"Processing content from {url} in multiple formats...")
    
    try:
        # Check if it's actually a URL or just a text string
        import re
        if not re.search(r'\.(gov|org|com|net|int|eu|info|edu|mil|[a-z]{2})(/|$)', url) and not url.startswith(('http://', 'https://')):
            logger.warning(f"Not a valid website URL: {url}")
            # For cases like "ACEA Regulatory Guide 2023" which isn't a URL
            if "ACEA" in url:
                # Fall back to the actual URL for ACEA
                url = "https://www.acea.auto/publications/"
                logger.info(f"Using fallback URL for ACEA: {url}")
            else:
                return {}  # Return empty result for non-URL strings
        
        # Use session with proper headers
        session = requests.Session()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        
        # Clean the URL more thoroughly to remove invisible characters
        url = clean_url(url)
        
        # Try alternative URLs if needed
        urls_to_try = [
            url,
            url.replace("https://www.", "https://"),
            "https://www." + url.replace("https://", "") if url.startswith("https://") else url
        ]
        
        # Add NHTSA specific alternatives if needed
        if "nhtsa.gov" in url:
            if "/laws-regulations/" not in url:
                urls_to_try.append("https://www.nhtsa.gov/laws-regulations")
                urls_to_try.append("https://www.nhtsa.gov/laws-regulations/fmvss")
                
            # NHTSA interpretations file - contains most of the legal interpretations
            urls_to_try.append("https://www.nhtsa.gov/nhtsa-interpretation-file-search")
            
            # For fuel type questions specifically
            if any(term in query.lower() for term in ["fuel", "gas", "gasoline", "diesel", "alternative"]):
                urls_to_try.append("https://www.nhtsa.gov/vehicle-manufacturers/cafe-fuel-economy")
        
        # Special handling for troublesome domains
        if "unece.org" in url:
            urls_to_try.append("https://unece.org/transport/vehicle-regulations-wp29")
            urls_to_try.append("https://unece.org/transport/standards/transport/vehicle-regulations-wp29")
        
        # Make sure URLs are unique
        urls_to_try = list(dict.fromkeys(urls_to_try))
        
        logger.info(f"URLs to try: {urls_to_try}")
        
        response = None
        successful_url = None
        content_source = ""
        
        for try_url in urls_to_try:
            try:
                logger.info(f"Attempting to access: {try_url}")
                response = session.get(try_url, headers=headers, timeout=60)  # Increased timeout for NHTSA
                
                # Check if successful
                if response.status_code == 200:
                    logger.info(f"Successfully accessed: {try_url}")
                    successful_url = try_url
                    content_source = "website"
                    break
                else:
                    logger.warning(f"Failed to access {try_url}: {response.status_code}")
            except requests.exceptions.RequestException as e:
                logger.error(f"Error accessing {try_url}: {str(e)}")
                continue
                
        if not response or response.status_code != 200:
            logger.error(f"Failed to access any URL variant for {url}")
            return {}
            
        # Detect content type
        content_type = response.headers.get('Content-Type', '').lower()
        logger.info(f"Content type: {content_type}")
        
        extracted_content = {}
        
        # Process based on content type
        if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
            # PDF content
            logger.info("Processing PDF content")
            pdf_file = io.BytesIO(response.content)
            
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
                
                if text.strip():
                    extracted_content[os.path.basename(successful_url)] = text
                    content_source = "pdf"
                    logger.info(f"Successfully processed PDF: extracted {len(text)} characters")
            except Exception as e:
                logger.error(f"Error processing PDF: {str(e)}")
        
        elif 'text/html' in content_type:
            # HTML content
            logger.info("Processing HTML content")
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements that may interfere with content extraction
            for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                script.decompose()
            
            # Look for main content areas
            main_content = ""
            
            # Try to find main content elements
            content_containers = soup.select("main, article, .content, #content, .main-content, .entry-content, .post-content")
            
            if content_containers:
                logger.info(f"Found {len(content_containers)} content containers")
                for container in content_containers:
                    container_text = container.get_text(separator='\n', strip=True)
                    if len(container_text) > len(main_content):
                        main_content = container_text
            else:
                # If no content containers found, extract from body with more careful processing
                logger.info("No content containers found, extracting from body")
                body = soup.find('body')
                if body:
                    # Remove common navigation and footer elements
                    for elem in body.select('nav, footer, aside, .navigation, .menu, .sidebar, .widget, .comment'):
                        elem.decompose()
                    
                    # Get all paragraphs and headings
                    content_elements = body.select('p, h1, h2, h3, h4, h5, h6, .paragraph, li, td, th, dl, dt, dd')
                    content_texts = []
                    
                    for elem in content_elements:
                        text = elem.get_text(strip=True)
                        if text and len(text) > 10:  # Ignore very small fragments
                            content_texts.append(text)
                    
                    main_content = '\n\n'.join(content_texts)
                    
                    # If still empty, try getting all visible text from body
                    if not main_content:
                        main_content = body.get_text(separator='\n', strip=True)
            
            if main_content:
                # Remove excessive whitespace
                main_content = re.sub(r'\n\s*\n', '\n\n', main_content)
                extracted_content[f"HTML content from {os.path.basename(successful_url)}"] = main_content
                content_source = "html"
                logger.info(f"Successfully extracted HTML content: {len(main_content)} characters")
            else:
                logger.warning("Failed to extract meaningful content from HTML")
        
        elif 'application/xml' in content_type or 'text/xml' in content_type or url.lower().endswith('.xml'):
            # XML content
            logger.info("Processing XML content")
            try:
                soup = BeautifulSoup(response.text, 'xml')
                xml_text = soup.get_text(separator='\n', strip=True)
                
                if xml_text:
                    extracted_content[f"XML content from {os.path.basename(successful_url)}"] = xml_text
                    content_source = "xml"
                    logger.info(f"Successfully processed XML: extracted {len(xml_text)} characters")
            except Exception as e:
                logger.error(f"Error processing XML: {str(e)}")
        
        elif 'application/json' in content_type or url.lower().endswith('.json'):
            # JSON content
            logger.info("Processing JSON content")
            try:
                json_data = response.json()
                json_text = json.dumps(json_data, indent=2)
                
                if json_text:
                    extracted_content[f"JSON content from {os.path.basename(successful_url)}"] = json_text
                    content_source = "json"
                    logger.info(f"Successfully processed JSON: extracted {len(json_text)} characters")
            except Exception as e:
                logger.error(f"Error processing JSON: {str(e)}")
        
        else:
            # Plain text or other content
            logger.info(f"Processing as plain text (content type: {content_type})")
            text = response.text
            
            if text:
                extracted_content[f"Content from {os.path.basename(successful_url)}"] = text
                content_source = "text"
                logger.info(f"Successfully processed text content: {len(text)} characters")
        
        # If no content was extracted, try as plain text
        if not extracted_content and response.text:
            logger.info("Falling back to plain text extraction")
            extracted_content[f"Text from {os.path.basename(successful_url)}"] = response.text
            content_source = "fallback_text"
            logger.info(f"Used fallback text extraction: {len(response.text)} characters")
        
        # Look for links to other potentially relevant documents if content is sparse
        if len('\n'.join(extracted_content.values())) < 1000 and content_source == "html":
            logger.info("Content is sparse, looking for links to other documents")
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Keywords relevant to regulations
            regulation_keywords = [
                'regulation', 'regulatory', 'rule', 'law', 'compliance', 'standard', 'requirement',
                'statute', 'directive', 'guidance', 'policy', 'procedure', 'manual', 'document',
                'fmvss', 'nhtsa', 'certification', 'safety'
            ]
            
            doc_links = []
            for link in soup.find_all('a'):
                href = link.get('href')
                text = link.get_text().lower()
                
                if not href:
                    continue
                    
                # Make relative links absolute
                if not href.startswith(('http://', 'https://')):
                    if href.startswith('/'):
                        from urllib.parse import urlparse
                        parsed_url = urlparse(successful_url)
                        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
                        href = base_url + href
                    else:
                        if successful_url.endswith('/'):
                            href = successful_url + href
                        else:
                            href = successful_url + '/' + href
                
                # Check if link is relevant to regulations
                is_relevant = any(keyword in text.lower() for keyword in regulation_keywords)
                has_doc_extension = any(href.lower().endswith(ext) for ext in ['.pdf', '.doc', '.docx', '.txt', '.rtf', '.html'])
                
                if is_relevant or has_doc_extension:
                    doc_links.append((text, href))
            
            # Process top 3 most relevant document links
            for i, (link_text, link_href) in enumerate(doc_links[:3]):
                logger.info(f"Following relevant document link: {link_text} at {link_href}")
                try:
                    link_response = session.get(link_href, headers=headers, timeout=30)
                    
                    if link_response.status_code == 200:
                        link_content_type = link_response.headers.get('Content-Type', '').lower()
                        
                        # Extract based on content type
                        if 'application/pdf' in link_content_type or link_href.lower().endswith('.pdf'):
                            # PDF
                            pdf_file = io.BytesIO(link_response.content)
                            try:
                                reader = PyPDF2.PdfReader(pdf_file)
                                text = ""
                                max_pages = min(50, len(reader.pages))
                                
                                for i in range(max_pages):
                                    page_text = reader.pages[i].extract_text()
                                    if page_text:
                                        text += page_text + "\n\n"
                                
                                if text.strip():
                                    extracted_content[f"{link_text}"] = text
                                    logger.info(f"Successfully processed linked PDF: {link_text}, {len(text)} characters")
                            except Exception as e:
                                logger.error(f"Error processing linked PDF: {str(e)}")
                        elif 'text/html' in link_content_type:
                            # HTML
                            link_soup = BeautifulSoup(link_response.text, 'html.parser')
                            for script in link_soup(["script", "style", "nav", "footer", "header"]):
                                script.decompose()
                            
                            content = link_soup.get_text(separator='\n', strip=True)
                            if content:
                                extracted_content[f"{link_text}"] = content
                                logger.info(f"Successfully processed linked HTML: {link_text}, {len(content)} characters")
                    else:
                        logger.warning(f"Failed to access linked document {link_href}: {link_response.status_code}")
                except Exception as e:
                    logger.error(f"Error following document link {link_href}: {str(e)}")
        
        return extracted_content
        
    except Exception as e:
        logger.error(f"Error processing content from {url}: {str(e)}")
        return {}


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
    """Process a query using the improved agent."""
    logger.info(f"Processing query: {query}, market: {market or 'Auto-detect'}, source: {source or 'Auto-detect'}")
    
    results = {
        "query": query,
        "market": market,
        "source": source,
        "selected_url": "",
        "content_items": {},
        "final_answer": ""
    }
    
    # Step 1: Check if client is initialized
    if not client:
        logger.error("Groq client not initialized")
        results["final_answer"] = "I apologize, but I couldn't connect to the AI service. Please try again later."
        return results
    
    # Step 2: Determine market and source if not provided
    if not market:
        logger.info("Determining market...")
        # Use enhanced market detection instead of the basic get_market_and_source
        results["market"] = enhanced_market_detection(query, client)
        logger.info(f"Enhanced market detection result: {results['market']}")
    
    # If we have a market but no source, find the most relevant source for that market
    if results["market"] != "UNCLEAR" and not source:
        logger.info(f"Determining appropriate source for market: {results['market']}")
        # Find sources that match the market
        matching_sources = []
        
        # Check each source to see if it matches the market
        for source_name, url in REGULATORY_WEBSITES.items():
            if results["market"].lower() in source_name.lower():
                matching_sources.append(source_name)
        
        # For US queries about fuel, prioritize EPA or DOE sources
        if results["market"] == "US" and any(term in query.lower() for term in ["fuel", "gas", "alternative", "gasoline", "diesel", "ev", "electric"]):
            for source_name in matching_sources:
                if "EPA" in source_name or "Environmental" in source_name or "Energy" in source_name:
                    results["source"] = source_name
                    logger.info(f"Selected source for US fuel query: {source_name}")
                    break
        
        # If we haven't selected a source yet but have matching sources, use the first one
        if not results.get("source") and matching_sources:
            results["source"] = matching_sources[0]
            logger.info(f"Selected first matching source: {results['source']}")
        # If no matching sources, set to NONE to trigger fallback
        elif not matching_sources:
            results["source"] = "NONE"
    
    # If we still don't have a source, use LLM to determine the best source
    if not results.get("source") or results["source"] == "NONE":
        logger.info("Using LLM to determine best source...")
        
        # Create a formatted list of all available regulatory sources
        source_list = ""
        for s_name, s_url in REGULATORY_WEBSITES.items():
            source_list += f"- {s_name}: {s_url}\n"
        
        prompt = f"""
        Based on the following query about automotive regulations, determine which regulatory source would be most relevant to answer it.

        User query: "{query}"
        
        Available regulatory sources:
        {source_list}
        
        Return ONLY the exact name of the most relevant regulatory source from the list above.
        """
        
        try:
            # Call LLM to determine source
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100
            )
            
            determined_source = response.choices[0].message.content.strip()
            logger.info(f"LLM suggested source: {determined_source}")
            
            # Check if the LLM's suggested source exists in our list
            if determined_source in REGULATORY_WEBSITES:
                results["source"] = determined_source
            else:
                # Try to find a close match
                for key in REGULATORY_WEBSITES.keys():
                    if determined_source.lower() in key.lower() or key.lower() in determined_source.lower():
                        results["source"] = key
                        logger.info(f"Found close match for source: {key}")
                        break
                else:
                    logger.warning(f"LLM returned invalid source: {determined_source}")
                    results["source"] = "NONE"
                
        except Exception as e:
            logger.error(f"Error determining source with LLM: {str(e)}")
            results["source"] = "NONE"
    
    # Select URL based on source, or fall back to market-based selection
    if results["source"] and results["source"] != "NONE" and results["source"] in REGULATORY_WEBSITES:
        results["selected_url"] = REGULATORY_WEBSITES[results["source"]]
        logger.info(f"Selected URL based on source: {results['selected_url']}")
    elif results["market"] and results["market"] != "UNCLEAR":
        # Try to find any source for the market if we have one
        logger.info(f"No valid source found, looking for any source for market: {results['market']}")
        
        for source_name, url in REGULATORY_WEBSITES.items():
            if results["market"].lower() in source_name.lower():
                results["source"] = source_name
                results["selected_url"] = url
                logger.info(f"Found fallback source for market: {source_name}")
                break
        else:
            # If still no source, provide informative error
            results["final_answer"] = f"I couldn't determine a specific regulatory source for {results['market']} regulations on this topic. Please select a specific source and try again."
            return results
    else:
        # If all attempts fail, provide a helpful error message
        results["final_answer"] = "I couldn't determine which regulatory source would be most relevant for your query. Please specify a market (like US, EU, China) or a specific regulatory agency in your query."
        return results
    
    # Process content from the selected URL - now handling multiple formats, not just PDFs
    logger.info(f"Extracting content from {results['selected_url']}...")
    results["content_items"] = process_multi_format_content(results["selected_url"], query, client)
    
    # Check if we got any content
    if not results["content_items"]:
        logger.warning("No content could be extracted from the selected source")
        
        # Try one more source from the same market as fallback
        logger.info("Trying fallback source from the same market")
        tried_already = [results["source"]]
        
        for source_name, url in REGULATORY_WEBSITES.items():
            if source_name not in tried_already and (results["market"].lower() in source_name.lower() or "Global" in source_name):
                logger.info(f"Trying alternative source: {source_name}")
                results["source"] = source_name
                results["selected_url"] = url
                results["content_items"] = process_multi_format_content(url, query, client)
                
                if results["content_items"]:
                    logger.info(f"Successfully extracted content from fallback source")
                    break
                
                tried_already.append(source_name)
        
        # If still no content, provide informative error
        if not results["content_items"]:
            results["final_answer"] = "I couldn't find relevant content to answer your query in the regulatory sources I checked. This could be due to technical difficulties accessing the websites, or because the specific information isn't available in the sources I have access to."
            return results
    
    # Analyze content and generate answer with token limit management
    logger.info(f"Analyzing content from {len(results['content_items'])} items...")
    results["final_answer"] = analyze_content_with_token_management(query, results["content_items"], client)
    
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
