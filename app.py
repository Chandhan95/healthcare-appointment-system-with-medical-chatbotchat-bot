from flask import Flask, render_template, request, jsonify
import requests
import json
from datetime import datetime

app = Flask(__name__)

# Medical Chatbot Class (Object-Oriented Paradigm)
class MedicalChatbot:
    def __init__(self, model_name="llama3.2:3b"):  # Changed to a more common model
        self.model = model_name
        self.ollama_url = "http://localhost:11434/api/generate"
        self.conversation_history = []
    
    def test_ollama_connection(self):
        """Test if Ollama is running and accessible"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def process_medical_query(self, user_input):
        """Process medical query with Ollama model"""
        # First test connection
        if not self.test_ollama_connection():
            return "❌ Cannot connect to Ollama server. Please make sure Ollama is running with 'ollama serve'"
        
        try:
            # Create medical-focused prompt
            medical_prompt = f"""You are a helpful medical information assistant. Please provide informative, accurate responses about health topics, but always remind users that this is for educational purposes only and they should consult healthcare professionals for medical advice.

User question: {user_input}

Please provide a helpful response and include a disclaimer about consulting healthcare professionals."""
            
            # Call Ollama API
            payload = {
                "model": self.model,
                "prompt": medical_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7
                }
            }
            
            print(f"🔄 Sending request to Ollama with model: {self.model}")
            
            response = requests.post(
                self.ollama_url, 
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=60  # Increased timeout
            )
            
            print(f"📡 Ollama response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result.get('response', '')
                
                if not ai_response:
                    return "⚠️ Received empty response from AI model. Try asking your question differently."
                
                # Add to conversation history
                self.conversation_history.append({
                    'user': user_input,
                    'bot': ai_response,
                    'timestamp': datetime.now().isoformat()
                })
                
                return ai_response
            else:
                error_text = response.text if response.text else "Unknown error"
                print(f"❌ Ollama error: {error_text}")
                return f"🔧 Ollama server error (Status {response.status_code}). Make sure the model '{self.model}' is installed with: ollama pull {self.model}"
                
        except requests.exceptions.Timeout:
            return "⏱️ Request timed out. The AI model might be loading. Please wait a moment and try again."
        except requests.exceptions.ConnectionError:
            return "🔌 Cannot connect to Ollama server. Please ensure Ollama is running with 'ollama serve'"
        except requests.exceptions.RequestException as e:
            return f"🚫 Network error: {str(e)}"
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            return "⚠️ An unexpected error occurred. Please try again."
    
    def get_conversation_history(self):
        """Get chat history"""
        return self.conversation_history

# Initialize chatbot
medical_bot = MedicalChatbot()

# Routes (Event-Driven Paradigm)
@app.route('/')
def index():
    """Render main chat interface"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages from frontend"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        print(f"💬 User message: {user_message}")
        
        # Process with medical chatbot
        bot_response = medical_bot.process_medical_query(user_message)
        
        print(f"🤖 Bot response: {bot_response[:100]}...")
        
        return jsonify({
            'response': bot_response,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Server error: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/history')
def get_history():
    """Get conversation history"""
    history = medical_bot.get_conversation_history()
    return jsonify(history)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        # Test Ollama connection
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        ollama_status = "connected" if response.status_code == 200 else "disconnected"
        
        # Try to get available models
        models = []
        if response.status_code == 200:
            models_data = response.json()
            models = [model.get('name', '') for model in models_data.get('models', [])]
            
    except Exception as e:
        ollama_status = "disconnected"
        models = []
    
    return jsonify({
        'status': 'healthy',
        'ollama': ollama_status,
        'available_models': models,
        'current_model': medical_bot.model,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/test-ollama')
def test_ollama():
    """Test endpoint to check Ollama connection"""
    try:
        # Test basic connection
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        
        if response.status_code == 200:
            models_data = response.json()
            models = [model.get('name', '') for model in models_data.get('models', [])]
            
            return jsonify({
                'status': 'success',
                'message': 'Ollama is running and accessible',
                'available_models': models,
                'current_model': medical_bot.model,
                'model_available': medical_bot.model in models
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f'Ollama returned status {response.status_code}'
            })
            
    except requests.exceptions.ConnectionError:
        return jsonify({
            'status': 'error',
            'message': 'Cannot connect to Ollama. Make sure it\'s running with: ollama serve'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error testing Ollama: {str(e)}'
        })

if __name__ == '__main__':
    print("🏥 Medical Chatbot Starting...")
    print("📋 Checking Ollama connection...")
    
    if medical_bot.test_ollama_connection():
        print("✅ Ollama is running and accessible")
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            models_data = response.json()
            models = [model.get('name', '') for model in models_data.get('models', [])]
            print(f"📚 Available models: {models}")
            
            if medical_bot.model not in models:
                print(f"⚠️  Model '{medical_bot.model}' not found!")
                print(f"💡 Install it with: ollama pull {medical_bot.model}")
            else:
                print(f"✅ Model '{medical_bot.model}' is available")
                
        except Exception as e:
            print(f"❌ Error checking models: {e}")
    else:
        print("❌ Ollama is not running or not accessible")
        print("💡 Start it with: ollama serve")
    
    print("🌐 Starting Flask server...")
    print("📱 Access at: http://localhost:5000")
    print("🔧 Test Ollama at: http://localhost:5000/test-ollama")
    
    app.run(debug=True, host='0.0.0.0', port=5000)