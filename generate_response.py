import os
import sys
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import firebase_admin
from firebase_admin import credentials, db
from firebase_admin import credentials

FIREBASE_CRED_PATH = "ai-clone-47beb-firebase-adminsdk-nrs6e-f5d7911fac.json"
cred = credentials.Certificate(FIREBASE_CRED_PATH)
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://ai-clone-47beb-default-rtdb.firebaseio.com"
})

if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_CRED_PATH)
    firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})

# Fetch the trained model path from Firebase
def get_model_path():
    print("Fetching model path from Firebase...")
    ref = db.reference("userResponses")
    data = ref.order_by_key().limit_to_last(1).get()
    if not data:
        print("Error: No user data found in Firebase.")
        return None
    user_id, user_data = next(iter(data.items()))
    model_path = user_data.get("modelPath", "gpt2")  # Fallback to base GPT-2
    print(f"Model path fetched: {model_path}")
    return model_path

# CLI arguments
if len(sys.argv) < 2:
    print("Usage: python generate_response.py <input_text>")
    sys.exit(1)

input_text = sys.argv[1]

# Fetch and validate model path
model_path = get_model_path()
if not model_path:
    print("Error: Unable to determine model path.")
    sys.exit(1)

# Load the trained model and tokenizer
print(f"Loading model from: {model_path}")
try:
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    sys.exit(1)

# Generate a response
try:
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input_ids, max_length=50, num_return_sequences=1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print(response)
except Exception as e:
    print(f"Error during response generation: {e}")
