import os
import tempfile
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import requests
import traceback
import time
from pdf_processor import PDFProcessor
import asyncio
import concurrent.futures
from functools import lru_cache

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure CORS - allow your GoDaddy domain
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://semicom-consultancy.com",  # Your GoDaddy domain
            "https://semicom-consultancy.com", # HTTPS version
            "http://localhost:5500",           # For local testing
            "http://127.0.0.1:5500",           # For local testing
            "*"                                # Allow all for testing
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Initialize PDF processor with caching
print("Initializing PDF processor...")
pdf_processor = None
try:
    pdf_processor = PDFProcessor()
    print("PDF processor initialized successfully")
    
    # AUTO-LOAD PDFs from data folder on startup
    print("\n=== AUTO-LOADING PDFs FROM DATA FOLDER ===")
    data_folder = "data"
    if os.path.exists(data_folder) and os.path.isdir(data_folder):
        pdf_files = [f for f in os.listdir(data_folder) if f.lower().endswith('.pdf')]
        print(f"Found {len(pdf_files)} PDF files in {data_folder}/")
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(data_folder, pdf_file)
            try:
                print(f"Processing: {pdf_file}")
                pdf_processor.process_pdf(pdf_path)
                print(f"✓ Successfully loaded: {pdf_file}")
            except Exception as e:
                print(f"✗ Failed to load {pdf_file}: {str(e)}")
    else:
        print(f"Data folder '{data_folder}' not found. Creating it...")
        os.makedirs(data_folder, exist_ok=True)
        
    # Get document count
    if pdf_processor.collection:
        doc_count = pdf_processor.collection.count()
        print(f"\n✅ Total documents in database: {doc_count}")
    
except Exception as e:
    print(f"Error initializing PDF processor: {str(e)}")
    pdf_processor = None

# NVIDIA API configuration
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "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")
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# Create a thread pool for parallel processing
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Cache for common queries
query_cache = {}
MAX_CACHE_SIZE = 100
CACHE_TTL_SECONDS = 300  # 5 minutes

def clean_cache():
    """Remove old cache entries"""
    current_time = time.time()
    expired_keys = []
    for key, (timestamp, _) in query_cache.items():
        if current_time - timestamp > CACHE_TTL_SECONDS:
            expired_keys.append(key)
        if len(query_cache) > MAX_CACHE_SIZE:
            # Remove oldest entries
            oldest_key = min(query_cache.keys(), key=lambda k: query_cache[k][0])
            expired_keys.append(oldest_key)
    
    for key in expired_keys:
        query_cache.pop(key, None)

# Simple greeting cache
GREETING_RESPONSES = [
    "Hello! I'm SERA, your Semicom assistant. How can I help you today?",
    "Hi there! SERA here, ready to assist with your Semicom inquiries.",
    "Greetings! I'm SERA from Semicom. What can I help you with?",
    "Welcome! I'm SERA, your Semicom expert. How may I assist you?",
    "Hello! I'm SERA, the Semicom customer service assistant. How can I help?"
]

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get chatbot status and document info"""
    try:
        if not pdf_processor:
            return jsonify({
                "status": "pdf_processor_not_ready",
                "message": "PDF processor is not initialized"
            }), 500
        
        collection = pdf_processor.get_collection()
        doc_count = collection.count() if collection else 0
        
        return jsonify({
            "status": "ready",
            "documents_loaded": doc_count,
            "nvidia_api": "configured" if NVIDIA_API_KEY else "not_configured",
            "backend": "running",
            "chatbot_name": "SERA",
            "cache_size": len(query_cache)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/')
def home():
    """Home page for Render health check"""
    return jsonify({
        "status": "online",
        "service": "RAG Chatbot API",
        "chatbot_name": "SERA",
        "endpoints": {
            "health": "/api/health (GET)",
            "upload": "/api/upload-pdf (POST)",
            "chat": "/api/chat (POST)"
        }
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "chatbot_name": "SERA",
        "pdf_processor": "ready" if pdf_processor else "not_ready",
        "nvidia_api": "configured" if NVIDIA_API_KEY else "not_configured",
        "cache_size": len(query_cache)
    }), 200

@app.route('/api/upload-pdf', methods=['POST', 'OPTIONS'])
def upload_pdf():
    """Endpoint to upload and process PDF files"""
    if request.method == 'OPTIONS':
        return '', 200
    
    if 'pdf' not in request.files:
        return jsonify({"error": "No PDF file provided"}), 400
    
    pdf_file = request.files['pdf']
    if pdf_file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if not pdf_file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "File must be a PDF"}), 400
    
    # Check file size (limit to 10MB)
    pdf_file.seek(0, 2)  # Seek to end
    file_size = pdf_file.tell()
    pdf_file.seek(0)  # Reset to beginning
    
    if file_size > 10 * 1024 * 1024:  # 10MB
        return jsonify({"error": "File too large. Maximum size is 10MB"}), 400
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        pdf_file.save(tmp_file.name)
        tmp_path = tmp_file.name
    
    try:
        # Process the PDF
        if not pdf_processor:
            return jsonify({"error": "PDF processor not ready"}), 500
        
        pdf_processor.process_pdf(tmp_path)
        
        # Get document count
        collection = pdf_processor.get_collection()
        doc_count = collection.count() if collection else 0
        
        return jsonify({
            "success": True,
            "message": "PDF processed successfully",
            "filename": pdf_file.filename,
            "documents_processed": doc_count,
            "chatbot_name": "SERA"
        }), 200
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return jsonify({"error": f"Failed to process PDF: {str(e)}"}), 500
    
    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass

def generate_response_fast(user_message):
    """Generate a fast response without LLM for common queries"""
    user_message_lower = user_message.lower().strip()
    
    # Quick greetings
    if any(greet in user_message_lower for greet in ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon']):
        import random
        return random.choice(GREETING_RESPONSES), False
    
    # Thanks responses
    if any(thanks in user_message_lower for thanks in ['thank', 'thanks', 'appreciate']):
        return "You're welcome! I'm glad I could help. Is there anything else you'd like to know about Semicom?", False
    
    # Farewells
    if any(bye in user_message_lower for bye in ['bye', 'goodbye', 'see you', 'farewell']):
        return "Goodbye! Feel free to reach out if you have more questions about Semicom. Have a great day!", False
    
    return None, False

@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat():
    """Main chat endpoint with RAG - Optimized for speed"""
    if request.method == 'OPTIONS':
        return '', 200
    
    start_time = time.time()
    
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        user_message = data.get('message', '').strip()
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        print(f"Received chat request: {user_message[:50]}...")
        
        # Clean old cache entries
        clean_cache()
        
        # Check for cached response
        cache_key = f"response:{user_message.lower()[:100]}"
        if cache_key in query_cache:
            cached_response, cached_has_context = query_cache[cache_key][1]
            print(f"Using cached response for: {user_message[:50]}...")
            return jsonify({
                "success": True,
                "response": cached_response,
                "has_context": cached_has_context,
                "chatbot_name": "SERA",
                "cached": True,
                "response_time_ms": int((time.time() - start_time) * 1000)
            })
        
        # Try fast response first
        fast_response, has_context = generate_response_fast(user_message)
        if fast_response:
            query_cache[cache_key] = (time.time(), (fast_response, has_context))
            return jsonify({
                "success": True,
                "response": fast_response,
                "has_context": has_context,
                "chatbot_name": "SERA",
                "cached": False,
                "fast_response": True,
                "response_time_ms": int((time.time() - start_time) * 1000)
            })
        
        # 1. Retrieve relevant context from PDFs (optimized)
        context = ""
        context_start = time.time()
        if pdf_processor:
            # Use a simpler, faster search for common queries
            user_message_lower = user_message.lower()
            
            # Optimize search based on query type
            if any(word in user_message_lower for word in ['contact', 'email', 'phone', 'address', 'location']):
                # Contact info queries - search with higher precision
                context = pdf_processor.fast_search(user_message, top_k=2)
            elif len(user_message.split()) < 5:  # Short queries
                context = pdf_processor.search_context(user_message, top_k=2)  # Reduced from 3
            else:
                # Complex queries - use normal search
                context = pdf_processor.search_context(user_message, top_k=2)
        
        context_time = time.time() - context_start
        
        # 2. Prepare the prompt with context
        # Check if user is asking about the chatbot's name
        user_message_lower = user_message.lower()
        is_asking_name = any(phrase in user_message_lower for phrase in [
            'who are you', 'what is your name', 'your name', 'what should i call you',
            'what are you', 'what\'s your name', 'whats your name', 'introduce yourself',
            'tell me about yourself'
        ])
        
        # Prepare optimized prompts
        if is_asking_name:
            # Use a fixed name response for speed
            name_response = "I'm SERA (Semicom Expert Response Assistant), your dedicated customer service assistant for Semicom. I'm here to help you with any questions about our semiconductor testing equipment, services, or company information. How can I assist you today?"
            query_cache[cache_key] = (time.time(), (name_response, False))
            return jsonify({
                "success": True,
                "response": name_response,
                "has_context": False,
                "chatbot_name": "SERA",
                "cached": False,
                "response_time_ms": int((time.time() - start_time) * 1000)
            })
        
        # Create optimized prompt
        prompt = ""
        if context and context.strip():
            # Truncate context if too long to reduce token count
            if len(context) > 1500:
                context = context[:1500] + "... [truncated]"
            
            prompt = f"""Context: {context}

Question: {user_message}

Answer as SERA from Semicom:"""
        else:
            # Simplified prompt for no context
            prompt = f"""Question: {user_message}

Answer as SERA from Semicom:"""
        
        # 3. Call NVIDIA API with timeout and retry
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Optimize payload for faster response
        payload = {
            "model": "mistralai/mistral-large-3-675b-instruct-2512",
            "messages": [
                {
                    "role": "system",
                    "content": "You are SERA from Semicom. Respond in first person. Be concise. Answer based on context if available. Never mention 'SEMISHARE' - always use 'Semicom'. Max 3 sentences."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 300,  # Reduced from 1024 for faster response
            "temperature": 0.2,  # Lower temperature for faster, more consistent responses
            "top_p": 0.8,
            "stream": False
        }
        
        print(f"Context search: {context_time:.2f}s, Calling NVIDIA API...")
        api_start = time.time()
        
        # Use thread pool for non-blocking API call
        future = thread_pool.submit(requests.post, NVIDIA_API_URL, headers=headers, json=payload, timeout=15)
        try:
            response = future.result(timeout=20)
        except concurrent.futures.TimeoutError:
            # Fallback response if API times out
            fallback_response = "I'm working on getting that information for you. In the meantime, you can visit our website at semicom-consultancy.com or contact our support team directly for the most current information about Semicom products and services."
            query_cache[cache_key] = (time.time(), (fallback_response, bool(context)))
            return jsonify({
                "success": True,
                "response": fallback_response,
                "has_context": bool(context),
                "chatbot_name": "SERA",
                "cached": False,
                "timeout_fallback": True,
                "response_time_ms": int((time.time() - start_time) * 1000)
            })
        
        api_time = time.time() - api_start
        
        if response.status_code == 200:
            result = response.json()
            assistant_reply = result['choices'][0]['message']['content']
            
            # Additional safeguard: Replace any accidental SEMISHARE mentions with Semicom
            assistant_reply = assistant_reply.replace('SEMISHARE', 'Semicom').replace('SemiShare', 'Semicom')
            
            # Cache the response
            query_cache[cache_key] = (time.time(), (assistant_reply, bool(context and context.strip())))
            
            total_time = time.time() - start_time
            print(f"Total response time: {total_time:.2f}s (Context: {context_time:.2f}s, API: {api_time:.2f}s)")
            
            return jsonify({
                "success": True,
                "response": assistant_reply,
                "has_context": bool(context and context.strip()),
                "chatbot_name": "SERA",
                "cached": False,
                "response_time_ms": int(total_time * 1000),
                "timing": {
                    "context_ms": int(context_time * 1000),
                    "api_ms": int(api_time * 1000),
                    "total_ms": int(total_time * 1000)
                }
            })
        else:
            print(f"NVIDIA API error: {response.status_code}")
            # Fallback response
            fallback_response = "I'm currently experiencing some technical difficulties. For immediate assistance with Semicom products, please visit our website at semicom-consultancy.com or email us directly."
            return jsonify({
                "success": True,
                "response": fallback_response,
                "has_context": bool(context),
                "chatbot_name": "SERA",
                "cached": False,
                "api_error": True,
                "response_time_ms": int((time.time() - start_time) * 1000)
            })
            
    except requests.exceptions.Timeout:
        return jsonify({"error": "Request timed out. Please try again."}), 504
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Cannot connect to AI service. Please check your internet connection."}), 503
    except Exception as e:
        print(f"Unexpected error in chat endpoint: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/api/documents', methods=['GET'])
def get_documents():
    """Get information about processed documents"""
    try:
        if not pdf_processor:
            return jsonify({"error": "PDF processor not ready"}), 500
        
        collection = pdf_processor.get_collection()
        if not collection:
            return jsonify({"documents": [], "count": 0})
        
        # Get count and some metadata
        count = collection.count()
        
        # Get a few sample documents (first 5)
        results = collection.get(limit=min(5, count))
        
        return jsonify({
            "count": count,
            "documents": results.get('documents', [])[:3] if results else [],
            "chatbot_name": "SERA"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/clear_cache', methods=['POST'])
def clear_cache():
    """Clear the response cache"""
    global query_cache
    cache_size = len(query_cache)
    query_cache.clear()
    return jsonify({
        "success": True,
        "message": f"Cache cleared. Removed {cache_size} entries.",
        "chatbot_name": "SERA"
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)