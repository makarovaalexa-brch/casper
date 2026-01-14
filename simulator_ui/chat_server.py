from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
from pathlib import Path
import traceback

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv
load_dotenv()

from casper.agents.user_simulator import MovieLensLLMSimulator
from casper.agents.casper_agent import CASPERAgent

app = Flask(__name__)
CORS(app)

# Initialize components
movielens_path = project_root / "data" / "movielens"

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment")

if not movielens_path.exists():
    raise FileNotFoundError(f"MovieLens data not found at {movielens_path}")

# Available models for selection
AVAILABLE_MODELS = {
    "gpt-4o": "GPT-4o (Most Capable)",
    "gpt-4o-mini": "GPT-4o Mini (Fast & Efficient)",
    "gpt-4-turbo": "GPT-4 Turbo",
    "gpt-3.5-turbo": "GPT-3.5 Turbo (Budget)",
    "gpt-5-nano": "GPT-5 Nano (Experimental)"
}

DEFAULT_USER_MODEL = "gpt-5-nano"
DEFAULT_AGENT_MODEL = "gpt-4o-mini"

# Cache for simulators with different models
simulators = {}

def get_simulator(model_name: str = None):
    """Get or create simulator with specified model."""
    if model_name is None:
        model_name = DEFAULT_USER_MODEL

    if model_name not in simulators:
        print(f"Creating simulator with model: {model_name}")
        simulators[model_name] = MovieLensLLMSimulator(
            str(movielens_path),
            model_name=model_name
        )
        print(f"Initialized user simulator with {len(simulators[model_name].user_profiles)} profiles")

    return simulators[model_name]

# Initialize default simulator
simulator = get_simulator(DEFAULT_USER_MODEL)

# Note: CASPER agent initialization will depend on actual agent implementation
# For now, keeping placeholder
# casper_agent = CASPERAgent(movielens_data_path=str(movielens_path))

@app.route('/')
def index():
    return send_from_directory('.', 'chat_ui.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_id = data.get('user_id')
        message = data.get('message', '')
        history = data.get('history', [])
        user_model = data.get('user_model', DEFAULT_USER_MODEL)
        agent_model = data.get('agent_model', DEFAULT_AGENT_MODEL)

        if not message:
            return jsonify({'error': 'Message is required'}), 400

        # Get simulator with specified model
        sim = get_simulator(user_model)

        # Get user profile
        if user_id and user_id != 'random':
            # Find specific user profile by ID
            user_profile = None
            for profile in sim.user_profiles:
                if str(profile['user_id']) == str(user_id):
                    user_profile = profile
                    break

            if not user_profile:
                return jsonify({
                    'error': f'User ID {user_id} not found',
                    'status': 'error'
                }), 404
        else:
            # Random user
            user_profile = sim.sample_user()

        # Build conversation history from chat history
        # Format: ["Agent: ...", "User: ...", ...]
        conversation_history = []
        for msg in history:
            if msg.get('isUser'):
                conversation_history.append(f"User: {msg.get('content', '')}")
            else:
                conversation_history.append(f"Agent: {msg.get('content', '')}")

        # Add current message
        conversation_history.append(f"Agent: {message}")

        # Generate response with full conversation context
        response = sim.simulate_response(
            message,
            user_profile,
            conversation_history=conversation_history if conversation_history else None
        )

        return jsonify({
            'response': response,
            'user_id': user_profile['user_id'],
            'user_model': user_model,
            'agent_model': agent_model,
            'status': 'success'
        })

    except Exception as e:
        # Preserve full exception details
        error_details = {
            'error': str(e),
            'type': type(e).__name__,
            'traceback': traceback.format_exc(),
            'status': 'error'
        }
        print(f"Error in /api/chat: {error_details['traceback']}")
        return jsonify(error_details), 500

@app.route('/api/users', methods=['GET'])
def get_users():
    """Get available user IDs from simulator"""
    try:
        user_ids = [profile['user_id'] for profile in simulator.user_profiles]
        return jsonify({
            'user_ids': user_ids,
            'count': len(user_ids),
            'status': 'success'
        })
    except Exception as e:
        error_details = {
            'error': str(e),
            'type': type(e).__name__,
            'traceback': traceback.format_exc(),
            'status': 'error'
        }
        print(f"Error in /api/users: {error_details['traceback']}")
        return jsonify(error_details), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models for user simulator and agent"""
    try:
        return jsonify({
            'models': AVAILABLE_MODELS,
            'defaults': {
                'user_model': DEFAULT_USER_MODEL,
                'agent_model': DEFAULT_AGENT_MODEL
            },
            'status': 'success'
        })
    except Exception as e:
        error_details = {
            'error': str(e),
            'type': type(e).__name__,
            'traceback': traceback.format_exc(),
            'status': 'error'
        }
        print(f"Error in /api/models: {error_details['traceback']}")
        return jsonify(error_details), 500

if __name__ == '__main__':
    print("\nCASPER Chat Server")
    print("Server: http://localhost:5000")
    print(f"User profiles: {len(simulator.user_profiles)}")
    print()
    app.run(debug=True, port=5000)
