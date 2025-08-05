import streamlit as st
import requests
import json
from image_generator import ImageGenerator
import speech_recognition as sr
import io
import base64
from datetime import datetime, date
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
import pandas as pd

# Try to download NLTK data (only if not already present)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
    except:
        pass

# Configure page
st.set_page_config(
    page_title="Enhanced Exam Assistant Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling - FIXED VERSION
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        margin-bottom: 30px;
    }
    
    .feature-button {
        display: inline-block;
        margin: 10px;
        padding: 15px 30px;
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-decoration: none;
        border-radius: 10px;
        font-weight: bold;
        transition: transform 0.3s ease;
    }
    
    .feature-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* FIXED: Made chat container transparent */
    .chat-container {
        background: transparent !important;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        max-height: 400px;
        overflow-y: auto;
        border: none !important;
    }
    
    .user-message {
        background: #007bff;
        color: white;
        padding: 10px 15px;
        border-radius: 18px;
        margin: 5px 0;
        margin-left: 20%;
        text-align: right;
        word-wrap: break-word;
    }
    
    .bot-message {
        background: rgba(233, 236, 239, 0.9);
        color: #333;
        padding: 10px 15px;
        border-radius: 18px;
        margin: 5px 0;
        margin-right: 20%;
        word-wrap: break-word;
        backdrop-filter: blur(10px);
    }
    
    .file-message {
        background: #28a745;
        color: white;
        padding: 8px 12px;
        border-radius: 15px;
        margin: 3px 0;
        margin-left: 25%;
        text-align: right;
        font-size: 0.9em;
    }
    
    .input-container {
        position: sticky;
        bottom: 0;
        background: transparent;
        padding: 15px 0;
        margin-top: 20px;
    }
    
    .chat-input-row {
        display: flex;
        gap: 10px;
        align-items: flex-end;
        background: rgba(255, 255, 255, 0.95);
        padding: 15px;
        border-radius: 15px;
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }
    
    .structured-content {
        background: rgba(248, 249, 250, 0.9);
        border-left: 4px solid #007bff;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
        backdrop-filter: blur(10px);
    }
    
    .content-section {
        margin: 10px 0;
        padding: 10px;
        border-radius: 5px;
    }
    
    .definition-section {
        background: rgba(227, 242, 253, 0.9);
        border-left: 3px solid #2196f3;
    }
    
    .advantages-section {
        background: rgba(232, 245, 232, 0.9);
        border-left: 3px solid #4caf50;
    }
    
    .disadvantages-section {
        background: rgba(255, 235, 238, 0.9);
        border-left: 3px solid #f44336;
    }
    
    .applications-section {
        background: rgba(255, 243, 224, 0.9);
        border-left: 3px solid #ff9800;
    }
    
    .history-section {
        background: rgba(243, 229, 245, 0.9);
        border-left: 3px solid #9c27b0;
    }
    
    .file-upload-section {
        margin-bottom: 10px;
    }
    
    .uploaded-file-tag {
        display: inline-block;
        background: #007bff;
        color: white;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        margin: 2px;
    }
    
    .chat-history-item {
        padding: 10px;
        margin: 5px 0;
        background: rgba(248, 249, 250, 0.9);
        border-radius: 8px;
        border-left: 3px solid #007bff;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    
    .chat-history-item:hover {
        background: rgba(233, 236, 239, 0.9);
    }
    
    .chat-history-title {
        font-weight: bold;
        font-size: 0.9em;
        color: #333;
        margin-bottom: 5px;
    }
    
    .chat-history-preview {
        font-size: 0.8em;
        color: #666;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    
    .chat-history-time {
        font-size: 0.7em;
        color: #999;
        margin-top: 5px;
    }
    
    .stFileUploader {
        display: none !important;
    }
    
    .upload-btn {
        background: #6c757d;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 12px;
        cursor: pointer;
        font-size: 0.9em;
        transition: background-color 0.3s;
    }
    
    .upload-btn:hover {
        background: #5a6268;
    }
    
    .image-gen-container {
        background: rgba(248, 249, 250, 0.9);
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        border: 2px solid #e9ecef;
        backdrop-filter: blur(10px);
    }
    
    .download-section {
        margin: 15px 0;
        padding: 10px;
        background: rgba(232, 244, 253, 0.9);
        border-radius: 10px;
        border-left: 4px solid #007bff;
    }
    
    .image-info-card {
        background: rgba(212, 237, 218, 0.9);
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 10px;
        margin: 10px 0;
    }
    
    .generation-params {
        background: rgba(255, 243, 205, 0.9);
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 10px;
        margin: 10px 0;
        font-size: 0.9em;
    }
    
    /* Fix for markdown rendering */
    .stMarkdown {
        color: inherit;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state - FIXED VERSION
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_feature' not in st.session_state:
    st.session_state.current_feature = None
if 'chat_sessions' not in st.session_state:
    st.session_state.chat_sessions = []
if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'show_upload_dialog' not in st.session_state:
    st.session_state.show_upload_dialog = False
if 'input_key' not in st.session_state:
    st.session_state.input_key = 0

# Cohere API configuration
COHERE_API_KEY = "7Inil0a76TZkMeSnu63ZAlW7f7HYQH1Iduuca8Kv"
COHERE_API_URL = "https://api.cohere.ai/v1/generate"

# Enhanced knowledge base for educational content
EDUCATIONAL_TOPICS = {
    "machine learning": {
        "definition": "Machine Learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
        "advantages": [
            "Automated decision making",
            "Handles large datasets efficiently",
            "Continuous improvement through learning",
            "Pattern recognition capabilities",
            "Reduces human error"
        ],
        "disadvantages": [
            "Requires large amounts of data",
            "Can be biased based on training data",
            "Black box problem - difficult to interpret",
            "Computationally expensive",
            "Overfitting issues"
        ],
        "applications": [
            "Image and speech recognition",
            "Recommendation systems",
            "Medical diagnosis",
            "Autonomous vehicles",
            "Financial fraud detection",
            "Natural language processing"
        ],
        "history": "Machine Learning concept emerged in 1959 when Arthur Samuel coined the term. Key milestones include perceptrons (1957), neural networks (1980s), and deep learning revolution (2010s)."
    },
    "artificial intelligence": {
        "definition": "Artificial Intelligence is the simulation of human intelligence in machines that are programmed to think and learn like humans.",
        "advantages": [
            "24/7 availability",
            "Faster processing and decision making",
            "Handles dangerous tasks",
            "Reduces human error",
            "Consistent performance"
        ],
        "disadvantages": [
            "High development costs",
            "Job displacement concerns",
            "Lack of creativity and emotions",
            "Ethical concerns",
            "Dependency on technology"
        ],
        "applications": [
            "Virtual assistants",
            "Autonomous vehicles",
            "Medical diagnosis",
            "Game playing (Chess, Go)",
            "Robotics",
            "Financial trading"
        ],
        "history": "AI concept dates back to 1950 with Alan Turing's 'Computing Machinery and Intelligence'. Major developments include expert systems (1970s), machine learning boom (1990s), and current deep learning era."
    },
    "data science": {
        "definition": "Data Science is an interdisciplinary field that combines statistics, programming, and domain expertise to extract insights from structured and unstructured data.",
        "advantages": [
            "Data-driven decision making",
            "Predictive analytics capabilities",
            "Business intelligence insights",
            "Process optimization",
            "Risk assessment and management"
        ],
        "disadvantages": [
            "Data quality issues",
            "Privacy and security concerns",
            "Requires specialized skills",
            "Time-consuming data preparation",
            "Interpretation challenges"
        ],
        "applications": [
            "Business analytics and forecasting",
            "Healthcare analytics",
            "Marketing and customer segmentation",
            "Financial risk modeling",
            "Supply chain optimization",
            "Social media analysis"
        ],
        "history": "Term 'Data Science' was popularized in the 2000s, though statistical analysis has existed for centuries. Modern data science emerged with big data technologies in the 2010s."
    },
    "blockchain": {
        "definition": "Blockchain is a distributed ledger technology that maintains a continuously growing list of records (blocks) that are linked and secured using cryptography.",
        "advantages": [
            "Decentralized and transparent",
            "Immutable and tamper-proof",
            "Reduced intermediary costs",
            "Enhanced security",
            "Global accessibility"
        ],
        "disadvantages": [
            "High energy consumption",
            "Scalability issues",
            "Regulatory uncertainty",
            "Technical complexity",
            "Storage limitations"
        ],
        "applications": [
            "Cryptocurrency transactions",
            "Supply chain management",
            "Digital identity verification",
            "Smart contracts",
            "Voting systems",
            "Healthcare records"
        ],
        "history": "Blockchain concept was introduced in 2008 by Satoshi Nakamoto as the underlying technology for Bitcoin. First implemented in 2009."
    }
}

class EnhancedNLPProcessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        try:
            tokens = word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
            return tokens
        except:
            return text.split()
    
    def extract_educational_topics(self, text):
        """Extract educational topics from text"""
        tokens = self.preprocess_text(text)
        found_topics = []
        
        for topic in EDUCATIONAL_TOPICS.keys():
            topic_tokens = self.preprocess_text(topic)
            if any(token in tokens for token in topic_tokens):
                found_topics.append(topic)
        
        return found_topics
    
    def is_question(self, text):
        """Determine if text is a question"""
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'can', 'will', 'do', 'does', 'is', 'are']
        text_lower = text.lower()
        return (text.strip().endswith('?') or 
                any(text_lower.startswith(qw) for qw in question_words) or
                'explain' in text_lower or 'define' in text_lower)
    
    def extract_intent(self, text):
        """Extract user intent from text"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['date', 'today', 'time', 'current']):
            return 'datetime_query'
        elif any(word in text_lower for word in ['definition', 'define', 'what is', 'explain']):
            return 'definition_request'
        elif any(word in text_lower for word in ['advantage', 'benefit', 'pros']):
            return 'advantages_request'
        elif any(word in text_lower for word in ['disadvantage', 'drawback', 'cons']):
            return 'disadvantages_request'
        elif any(word in text_lower for word in ['application', 'use case', 'example']):
            return 'applications_request'
        elif any(word in text_lower for word in ['history', 'origin', 'development']):
            return 'history_request'
        else:
            return 'general_query'

def get_current_datetime_info():
    """Get current date and time information"""
    now = datetime.now()
    today = date.today()
    
    return {
        'current_date': today.strftime("%B %d, %Y"),
        'current_time': now.strftime("%I:%M %p"),
        'day_of_week': today.strftime("%A"),
        'month': today.strftime("%B"),
        'year': today.year
    }

def format_educational_content(topic, content_type='all'):
    """Format educational content in a structured way"""
    if topic not in EDUCATIONAL_TOPICS:
        return None
    
    topic_data = EDUCATIONAL_TOPICS[topic]
    formatted_content = f"## 📚 {topic.title()}\n\n"
    
    if content_type in ['all', 'definition']:
        formatted_content += f"### 🔍 Definition\n{topic_data['definition']}\n\n"
    
    if content_type in ['all', 'advantages']:
        formatted_content += "### ✅ Advantages\n"
        for adv in topic_data['advantages']:
            formatted_content += f"• {adv}\n"
        formatted_content += "\n"
    
    if content_type in ['all', 'disadvantages']:
        formatted_content += "### ❌ Disadvantages\n"
        for dis in topic_data['disadvantages']:
            formatted_content += f"• {dis}\n"
        formatted_content += "\n"
    
    if content_type in ['all', 'applications']:
        formatted_content += "### 🚀 Applications\n"
        for app in topic_data['applications']:
            formatted_content += f"• {app}\n"
        formatted_content += "\n"
    
    if content_type in ['all', 'history'] and 'history' in topic_data:
        formatted_content += f"### 📖 History\n{topic_data['history']}\n\n"
    
    return formatted_content

def get_enhanced_cohere_response(message, files_context=""):
    """Enhanced response generation with NLP processing"""
    nlp = EnhancedNLPProcessor()
    
    # Extract intent and topics
    intent = nlp.extract_intent(message)
    educational_topics = nlp.extract_educational_topics(message)
    
    # Handle datetime queries
    if intent == 'datetime_query':
        datetime_info = get_current_datetime_info()
        if 'date' in message.lower() or 'today' in message.lower():
            return f"📅 Today's date is {datetime_info['current_date']} ({datetime_info['day_of_week']})"
        elif 'time' in message.lower():
            return f"🕐 Current time is {datetime_info['current_time']}"
        else:
            return f"📅 Today is {datetime_info['day_of_week']}, {datetime_info['current_date']} and the current time is {datetime_info['current_time']}"
    
    # Handle educational content requests
    if educational_topics:
        response = ""
        for topic in educational_topics:
            if intent == 'definition_request':
                content = format_educational_content(topic, 'definition')
            elif intent == 'advantages_request':
                content = format_educational_content(topic, 'advantages')
            elif intent == 'disadvantages_request':
                content = format_educational_content(topic, 'disadvantages')
            elif intent == 'applications_request':
                content = format_educational_content(topic, 'applications')
            elif intent == 'history_request':
                content = format_educational_content(topic, 'history')
            else:
                content = format_educational_content(topic, 'all')
            
            if content:
                response += content + "\n"
        
        if response:
            return response
    
    # Fall back to Cohere API for other queries
    headers = {
        'Authorization': f'Bearer {COHERE_API_KEY}',
        'Content-Type': 'application/json',
    }
    
    context = f"\nFiles uploaded: {files_context}\n" if files_context else ""
    datetime_info = get_current_datetime_info()
    
    prompt = f"""You are an Enhanced Exam Assistant Chatbot designed to help students with comprehensive academic support. 
    Current date: {datetime_info['current_date']}
    Current time: {datetime_info['current_time']}
    
    When discussing educational topics, always provide:
    1. Clear definition
    2. Advantages and benefits
    3. Disadvantages and limitations
    4. Real-world applications
    5. Historical context (if relevant)
    
    {context}
    Student question: {message}
    
    Provide a comprehensive, structured response:"""
    
    data = {
        'model': 'command',
        'prompt': prompt,
        'max_tokens': 500,
        'temperature': 0.7,
        'k': 0,
        'stop_sequences': [],
        'return_likelihoods': 'NONE'
    }
    
    try:
        response = requests.post(COHERE_API_URL, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            return result['generations'][0]['text'].strip()
        else:
            return f"Sorry, I'm having trouble connecting right now. Error: {response.status_code}"
    except Exception as e:
        return f"Sorry, there was an error processing your request: {str(e)}"

def process_uploaded_file(uploaded_file):
    """Process uploaded file and return relevant information"""
    try:
        file_info = {
            'name': uploaded_file.name,
            'type': uploaded_file.type,
            'size': uploaded_file.size
        }
        
        # Read file content based on type
        if uploaded_file.type.startswith('text/'):
            content = str(uploaded_file.read(), "utf-8")
            file_info['content'] = content[:1000]  # First 1000 characters
        elif uploaded_file.type == 'application/pdf':
            file_info['content'] = "PDF file uploaded - content analysis available"
        elif uploaded_file.type.startswith('image/'):
            file_info['content'] = "Image file uploaded - visual analysis available"
        else:
            file_info['content'] = "File uploaded successfully"
        
        return file_info
    except Exception as e:
        return {'name': uploaded_file.name, 'error': str(e)}

# FIXED: Improved session management
def save_chat_session():
    """Save current chat session to history"""
    if st.session_state.chat_history and len(st.session_state.chat_history) > 0:
        session_id = st.session_state.current_session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get first user message for title
        first_user_message = None
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                first_user_message = msg['message']
                break
        
        if first_user_message:
            title = first_user_message[:50] + "..." if len(first_user_message) > 50 else first_user_message
            
            # Check if session already exists and update it
            session_exists = False
            for i, session in enumerate(st.session_state.chat_sessions):
                if session['id'] == session_id:
                    st.session_state.chat_sessions[i]['messages'] = st.session_state.chat_history.copy()
                    st.session_state.chat_sessions[i]['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M")
                    session_exists = True
                    break
            
            # If session doesn't exist, create new one
            if not session_exists:
                session = {
                    'id': session_id,
                    'title': title,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'messages': st.session_state.chat_history.copy()
                }
                st.session_state.chat_sessions.append(session)
            
            st.session_state.current_session_id = session_id

def load_chat_session(session_id):
    """Load a specific chat session"""
    for session in st.session_state.chat_sessions:
        if session['id'] == session_id:
            st.session_state.chat_history = session['messages'].copy()
            st.session_state.current_session_id = session_id
            st.rerun()

def clear_current_chat():
    """Clear current chat history"""
    # Save current chat before clearing if it has messages
    if st.session_state.chat_history:
        save_chat_session()
    
    st.session_state.chat_history = []
    st.session_state.current_session_id = None
    st.session_state.uploaded_files = []
    st.session_state.input_key += 1  # Clear input field

def build_professional_prompt(base_prompt, quality_preset, camera_style, detail_level, image_size):
    """Build a professional prompt optimized for sharp, detailed images"""
    
    # Extract size number from selection
    size_num = image_size.split('x')[0]
    
    # Quality-specific keywords
    quality_keywords = {
        "Standard": ["clear", "detailed"],
        "Professional": ["professional photography", "sharp focus", "high resolution", "studio quality"],
        "Ultra Sharp": ["ultra sharp", "crystal clear", "razor sharp focus", "professional DSLR", "8K quality", "hyperdetailed"],
        "Maximum Detail": ["maximum detail", "ultra high definition", "pixel perfect", "professional studio lighting", "tack sharp", "flawless clarity", "microscopic detail"]
    }
    
    # Camera-specific technical terms
    camera_keywords = {
        "Auto": ["sharp focus"],
        "DSLR Pro": ["shot with Canon 5D Mark IV", "85mm lens", "f/1.4 aperture", "professional DSLR", "bokeh"],
        "Medium Format": ["Hasselblad medium format", "Phase One", "ultra high resolution", "commercial photography"],
        "Cinema 8K": ["cinema camera", "RED 8K", "cinematic lighting", "film grain", "professional cinematography"],
        "Macro Lens": ["macro photography", "100mm macro lens", "extreme close-up", "microscopic detail", "ultra sharp macro"]
    }
    
    # Detail level keywords
    detail_keywords = {
        "Normal": ["clear details"],
        "High": ["high detail", "fine textures", "sharp edges"],
        "Ultra": ["ultra detailed", "intricate details", "fine grain", "texture detail"],
        "Hyperdetailed": ["hyperdetailed", "every pore visible", "skin texture", "fabric weave", "individual hair strands"]
    }
    
    # Technical quality terms
    technical_terms = [
        f"{size_num}K resolution",
        "RAW format",
        "professional lighting",
        "perfect exposure",
        "noise-free",
        "anti-aliasing",
        "ultra sharp focus",
        "perfect clarity"
    ]
    
    # Build enhanced prompt
    enhanced_parts = [base_prompt.strip()]
    enhanced_parts.extend(quality_keywords.get(quality_preset, []))
    enhanced_parts.extend(camera_keywords.get(camera_style, []))
    enhanced_parts.extend(detail_keywords.get(detail_level, []))
    enhanced_parts.extend(technical_terms[:4])  # Add first 4 technical terms
    
    return ", ".join(enhanced_parts)

# Sidebar for Chat History - FIXED VERSION
with st.sidebar:
    st.title("💬 Chat History")
    
    # Clear button
    if st.button("🗑️ Clear Current Chat", use_container_width=True, type="secondary"):
        clear_current_chat()
        st.rerun()
    
    # New Chat button
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        clear_current_chat()
        st.rerun()
    
    st.markdown("---")
    
    # Display current date and time
    datetime_info = get_current_datetime_info()
    st.info(f"📅 {datetime_info['current_date']}\n🕐 {datetime_info['current_time']}")
    
    st.markdown("---")
    
    # Display chat sessions - FIXED VERSION
    if st.session_state.chat_sessions:
        st.subheader("Previous Chats")
        for session in reversed(st.session_state.chat_sessions[-10:]):  # Show last 10 sessions
            with st.container():
                # Create a unique key for each button
                button_key = f"chat_btn_{session['id']}"
                
                if st.button(
                    f"🗨️ {session['title']}", 
                    key=button_key,
                    help=f"Started: {session['timestamp']}",
                    use_container_width=True
                ):
                    load_chat_session(session['id'])
                
                # Show timestamp
                st.caption(session['timestamp'])
                st.markdown("---")
    else:
        st.info("No previous chats yet. Start a conversation!")

# Header
st.markdown("<h1 class='main-header'>🎓 Enhanced Exam Assistant Chatbot</h1>", unsafe_allow_html=True)

# Feature showcase
st.markdown("### 🌟 Enhanced Features")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.info("🧠 **Smart NLP**\nIntelligent question understanding")

with col2:
    st.info("📚 **Educational Content**\nStructured topic explanations")

with col3:
    st.info("📅 **Real-time Info**\nCurrent date and time queries")

with col4:
    st.info("🎨 **Image Generation**\nAdvanced AI image creation")

# Feature button for Image Generator
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("🎨 Image Generator", use_container_width=True):
        st.session_state.current_feature = 'image'

# Feature-specific interface - Image Generator
if st.session_state.current_feature == 'image':
    st.subheader("🎨 Image Generator")
    try:
        image_gen = ImageGenerator()
        
        # Advanced image generation options
        st.markdown("### 🎯 **Anti-Blur Settings**")
        col1, col2 = st.columns([2, 1])
        with col1:
            prompt = st.text_area(
                "Enter detailed image description:", 
                placeholder="Be very specific: A professional portrait of a woman with sharp eyes, detailed skin texture, studio lighting, shot with Canon 5D Mark IV, 85mm lens, f/1.4, ultra-sharp focus, 8K resolution...",
                height=100
            )
        
        with col2:
            image_size = st.selectbox(
                "Resolution",
                options=["512x512 (Fast)", "768x768 (Good)", "1024x1024 (Better)", "1536x1536 (Best)", "2048x2048 (Ultra)"],
                index=3,  # Default to 1536x1536
                help="Higher resolution = much sharper images"
            )
        
        # Professional quality controls
        col3, col4, col5, col6 = st.columns(4)
        with col3:
            quality_preset = st.selectbox(
                "Quality Preset",
                options=["Standard", "Professional", "Ultra Sharp", "Maximum Detail"],
                index=2,
                help="Ultra Sharp adds professional photography keywords"
            )
        
        with col4:
            camera_style = st.selectbox(
                "Camera Style",
                options=["Auto", "DSLR Pro", "Medium Format", "Cinema 8K", "Macro Lens"],
                index=1,
                help="Simulates professional camera equipment"
            )
            
        with col5:
            detail_level = st.selectbox(
                "Detail Level",
                options=["Normal", "High", "Ultra", "Hyperdetailed"],
                index=2,
                help="More details = less blur"
            )
            
        with col6:
            post_processing = st.checkbox(
                "🔍 AI Sharpening",
                value=True,
                help="Apply advanced sharpening algorithms"
            )
        
        if st.button("🎯 Generate Ultra-Sharp Image", use_container_width=True, type="primary"):
            if prompt:
                # Generate professional prompt based on settings
                professional_prompt = build_professional_prompt(prompt, quality_preset, camera_style, detail_level, image_size)
                
                # Show the enhanced prompt
                with st.expander("🔍 View Enhanced Pro Prompt"):
                    st.code(professional_prompt, language="text")
                
                with st.spinner("Generating ultra-sharp professional image... Using advanced AI techniques..."):
                    try:
                        # Extract clean size for API
                        clean_size = image_size.split(' ')[0]  # Get "1536x1536" from "1536x1536 (Best)"
                        
                        # Check what parameters the generate_image method accepts
                        import inspect
                        sig = inspect.signature(image_gen.generate_image)
                        param_names = list(sig.parameters.keys())
                        
                        # Try to generate image with enhanced parameters if supported
                        if len(param_names) > 1 or any(param in param_names for param in ['size', 'quality', 'style']):
                            try:
                                # Method supports additional parameters
                                image_url = image_gen.generate_image(
                                    prompt=professional_prompt,
                                    size=clean_size,
                                    quality="ultra" if quality_preset in ["Ultra Sharp", "Maximum Detail"] else "hd",
                                    style="photographic"
                                )
                            except TypeError:
                                # Fallback to basic call with professional prompt
                                image_url = image_gen.generate_image(professional_prompt)
                        else:
                            # Method only accepts prompt - use professional prompt
                            image_url = image_gen.generate_image(professional_prompt)
                            st.info(f"🔧 Using professional prompt enhancement")
                            st.info(f"📊 Settings: {clean_size}, {quality_preset}, {camera_style}")
                        
                        if image_url:
                            # Display the generated image with no compression
                            st.image(
                                image_url, 
                                caption=f"Ultra-Sharp: {prompt[:50]}... ({clean_size}, {quality_preset})",
                                use_column_width=False  # Prevent Streamlit compression
                            )
                            
                            # Advanced post-processing if enabled
                            if post_processing:
                                try:
                                    import requests
                                    from PIL import Image, ImageEnhance, ImageFilter
                                    import io
                                    import numpy as np
                                    
                                    response = requests.get(image_url)
                                    if response.status_code == 200:
                                        # Advanced AI-like sharpening process
                                        img = Image.open(io.BytesIO(response.content))
                                        
                                        # Convert to RGB if needed
                                        if img.mode != 'RGB':
                                            img = img.convert('RGB')
                                        
                                        # Multi-stage sharpening process
                                        # Stage 1: Unsharp mask (professional technique)
                                        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
                                        
                                        # Stage 2: Enhanced sharpness
                                        enhancer = ImageEnhance.Sharpness(img)
                                        img = enhancer.enhance(1.3)
                                        
                                        # Stage 3: Contrast enhancement for clarity
                                        contrast_enhancer = ImageEnhance.Contrast(img)
                                        img = contrast_enhancer.enhance(1.1)
                                        
                                        # Stage 4: Detail enhancement
                                        img = img.filter(ImageFilter.DETAIL)
                                        
                                        # Convert back to bytes with maximum quality
                                        enhanced_buffer = io.BytesIO()
                                        img.save(enhanced_buffer, format='PNG', quality=100, optimize=False)
                                        enhanced_data = enhanced_buffer.getvalue()
                                        
                                        # Show enhanced image
                                        st.markdown("### 🔍 **AI Enhanced Version:**")
                                        st.image(
                                            enhanced_data,
                                            caption="AI Enhanced - Ultra Sharp with Professional Post-Processing",
                                            use_column_width=False
                                        )
                                        
                                        # Download buttons for both versions
                                        col_orig, col_enhanced = st.columns(2)
                                        
                                        with col_orig:
                                            st.download_button(
                                                label="📥 Download Original",
                                                data=response.content,
                                                file_name=f"original_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                                mime="image/png",
                                                use_container_width=True,
                                                type="secondary"
                                            )
                                        
                                        with col_enhanced:
                                            st.download_button(
                                                label="🎯 Download AI Enhanced",
                                                data=enhanced_data,
                                                file_name=f"ultra_sharp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                                mime="image/png",
                                                use_container_width=True,
                                                type="primary"
                                            )
                                        
                                        st.success("✨ **Multi-stage AI enhancement applied:**\n- Unsharp mask filtering\n- Professional sharpening\n- Contrast optimization\n- Detail enhancement")
                                        
                                except ImportError:
                                    st.warning("📦 Install Pillow for AI enhancement: `pip install Pillow`")
                                    # Fallback download
                                    try:
                                        import requests
                                        response = requests.get(image_url)
                                        if response.status_code == 200:
                                            st.download_button(
                                                label="📥 Download Image",
                                                data=response.content,
                                                file_name=f"sharp_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                                mime="image/png",
                                                use_container_width=True,
                                                type="primary"
                                            )
                                    except Exception:
                                        st.info("Right-click image to save manually.")
                                except Exception as enhance_error:
                                    st.warning(f"Enhancement failed: {enhance_error}")
                                    # Regular download
                                    try:
                                        import requests
                                        response = requests.get(image_url)
                                        if response.status_code == 200:
                                            st.download_button(
                                                label="📥 Download Image",
                                                data=response.content,
                                                file_name=f"professional_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                                mime="image/png",
                                                use_container_width=True,
                                                type="primary"
                                            )
                                    except Exception:
                                        st.info("Right-click image to save manually.")
                            else:
                                # Simple download without enhancement
                                try:
                                    import requests
                                    response = requests.get(image_url)
                                    if response.status_code == 200:
                                        st.download_button(
                                            label="📥 Download Professional Image",
                                            data=response.content,
                                            file_name=f"professional_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                            mime="image/png",
                                            use_container_width=True,
                                            type="primary"
                                        )
                                except Exception:
                                    st.info("Right-click image to save manually.")
                            
                            # Success message with technical details
                            st.success(f"🎯 **Ultra-sharp image generated!**\n- Resolution: {clean_size}\n- Quality: {quality_preset}\n- Camera: {camera_style}\n- Detail: {detail_level}")
                            
                            # Professional tips
                            with st.expander("🎓 **Pro Tips for Maximum Sharpness**"):
                                st.markdown(f"""
                                **Based on research from top AI image generators:**
                                
                                ✅ **Your Current Settings Analysis:**
                                - Resolution: {clean_size} {'✅ Excellent' if '1536' in clean_size or '2048' in clean_size else '⚠️ Try higher'}
                                - Quality: {quality_preset} {'✅ Professional' if quality_preset in ['Ultra Sharp', 'Maximum Detail'] else '💡 Try Ultra Sharp'}
                                - Camera: {camera_style} {'✅ Pro Equipment' if camera_style != 'Auto' else '💡 Try DSLR Pro'}
                                
                                **🔍 Prompt Writing Secrets:**
                                - Use structured prompts detailing subject, setting, composition, lighting, and camera settings. Adding terms like "8K resolution, ultra-sharp focus, cinematic depth" ensures high-quality professional results. The key to a great prompt is specificity.
                                - Add technical camera terms: "shot with Canon 5D Mark IV, 85mm lens, f/1.4"
                                - Include lighting details: "studio lighting, soft box, professional setup"
                                - Specify texture: "skin texture visible, fabric weave, individual hair strands"
                                
                                **🎯 Best Resolution Settings:**
                                - 1536x1536 or higher for maximum detail
                                - 4x resolution upscaling transforms images to up to 4K quality
                                - Square formats (1024x1024, 1536x1536) often produce sharper results
                                
                                **⚡ Quick Fixes for Blur:**
                                - Add "tack sharp", "razor sharp focus", "crystal clear"
                                - Include "professional studio photography"
                                - Use "hyperdetailed" for maximum detail
                                - Avoid words like "soft", "dreamy", "atmospheric"
                                """)
                        else:
                            st.error("❌ **Failed to generate image.**")
                            st.info("💡 **Troubleshooting tips:**\n- Try a simpler, more specific prompt\n- Check your internet connection\n- Ensure ImageGenerator API is working\n- Try different quality settings")
                            
                    except Exception as e:
                        st.error(f"Error generating image: {str(e)}")
                        st.info("💡 Tips for better results:\n- Be more specific in your description\n- Try simpler prompts\n- Check your internet connection")
            else:
                st.warning("⚠️ Please enter a description for the image you want to generate.")
    except Exception as e:
        st.error(f"Image generator not available: {str(e)}")
        st.info("💡 Make sure the ImageGenerator class is properly configured with API credentials.")

# Chat interface
st.markdown("---")
st.subheader("💬 Enhanced Chat Assistant")

# Display chat history with enhanced formatting - FIXED VERSION
chat_container = st.container()
with chat_container:
    if st.session_state.chat_history:
        for i, chat in enumerate(st.session_state.chat_history):
            # Create unique key for each message
            msg_key = f"msg_{i}_{chat['role']}"
            
            if chat['role'] == 'user':
                st.markdown(f"""
                <div class='user-message'>
                    {chat['message']}
                </div>
                """, unsafe_allow_html=True)
            elif chat['role'] == 'file':
                st.markdown(f"""
                <div class='file-message'>
                    {chat['message']}
                </div>
                """, unsafe_allow_html=True)
            else:
                # Enhanced bot message display with structured content
                if any(topic in chat['message'].lower() for topic in EDUCATIONAL_TOPICS.keys()):
                    # Use markdown for structured educational content
                    with st.container():
                        st.markdown(chat['message'])
                else:
                    st.markdown(f"""
                    <div class='bot-message'>
                        {chat['message']}
                    </div>
                    """, unsafe_allow_html=True)

# Chat input area with enhanced features
st.markdown('<div class="input-container">', unsafe_allow_html=True)

# Show uploaded files as tags if any
if st.session_state.uploaded_files:
    st.markdown('<div class="file-upload-section">', unsafe_allow_html=True)
    for file in st.session_state.uploaded_files:
        st.markdown(f'<span class="uploaded-file-tag">📎 {file.name}</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Upload dialog
if st.session_state.show_upload_dialog:
    with st.expander("📁 Upload Files", expanded=True):
        uploaded_files = st.file_uploader(
            "Choose files to upload", 
            accept_multiple_files=True,
            type=['txt', 'pdf', 'docx', 'jpg', 'jpeg', 'png', 'csv'],
            key=f"file_uploader_{st.session_state.input_key}"
        )
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("✅ Confirm Upload", type="primary", key=f"confirm_upload_{st.session_state.input_key}"):
                if uploaded_files:
                    st.session_state.uploaded_files = uploaded_files
                    st.session_state.show_upload_dialog = False
                    st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")
                    st.rerun()
                else:
                    st.warning("Please select files to upload.")
        
        with col_b:
            if st.button("❌ Cancel", key=f"cancel_upload_{st.session_state.input_key}"):
                st.session_state.show_upload_dialog = False
                st.rerun()

# Chat input row with improved styling
st.markdown('<div class="chat-input-row">', unsafe_allow_html=True)

# Create columns for upload, input and send buttons
col1, col2, col3 = st.columns([0.15, 0.65, 0.2])

# Upload button
with col1:
    if st.button("📁 Upload", use_container_width=True, type="secondary", key=f"upload_btn_{st.session_state.input_key}"):
        st.session_state.show_upload_dialog = True
        st.rerun()

# Text input with auto-message support
with col2:
    default_value = ""
    if hasattr(st.session_state, 'auto_message'):
        default_value = st.session_state.auto_message
        del st.session_state.auto_message
    
    user_input = st.text_input(
        "Type your message...", 
        key=f"chat_input_{st.session_state.input_key}", 
        label_visibility="collapsed",
        value=default_value
    )

# Send button
with col3:
    send_button = st.button("Send", use_container_width=True, type="primary", key=f"send_btn_{st.session_state.input_key}")

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Process user input with enhanced NLP - FIXED VERSION
if send_button and user_input:
    # Create new session ID if this is the first message
    if not st.session_state.current_session_id or len(st.session_state.chat_history) == 0:
        st.session_state.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Process uploaded files
    files_context = ""
    if st.session_state.uploaded_files:
        for uploaded_file in st.session_state.uploaded_files:
            file_info = process_uploaded_file(uploaded_file)
            files_context += f"{file_info['name']} ({file_info.get('type', 'unknown')}), "
            
            # Add file message to chat
            st.session_state.chat_history.append({
                'role': 'file', 
                'message': f"📎 Uploaded: {file_info['name']}"
            })
        
        # Clear uploaded files after processing
        st.session_state.uploaded_files = []
    
    # Add user message to chat history
    st.session_state.chat_history.append({'role': 'user', 'message': user_input})
    
    # Get enhanced bot response with NLP processing
    with st.spinner("🧠 Processing with enhanced AI..."):
        bot_response = get_enhanced_cohere_response(user_input, files_context)
    
    # Add bot response to chat history
    st.session_state.chat_history.append({'role': 'bot', 'message': bot_response})
    
    # Auto-save session after each message
    save_chat_session()
    
    # Clear input by incrementing the key
    st.session_state.input_key += 1
    
    # Rerun to update chat display
    st.rerun()

# Enhanced Footer with feature highlights
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🎓 Enhanced Exam Assistant Chatbot | Powered by Advanced AI & NLP</p>
    <p>✨ Features: Smart NLP Processing • Educational Content • Real-time Info • Image Generation</p>
</div>
""", unsafe_allow_html=True)