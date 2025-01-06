import os
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import firebase_admin
from firebase_admin import credentials, db
from datasets import Dataset
from firebase_admin import credentials

FIREBASE_CRED_PATH = "ai-clone-47beb-firebase-adminsdk-nrs6e-f5d7911fac.json"
cred = credentials.Certificate(FIREBASE_CRED_PATH)
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://ai-clone-47beb-default-rtdb.firebaseio.com"
})

if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_CRED_PATH)
    firebase_admin.initialize_app(cred, {
        "databaseURL": FIREBASE_DB_URL
    })

# Load user data from Firebase
def fetch_user_data():
    print("Fetching user data from Firebase...")
    ref = db.reference("userResponses")  # Reference the 'userResponses' node
    data = ref.order_by_key().limit_to_last(1).get()  # Fetch the most recent user data
    if not data:
        print("No user data found in Firebase.")
        return None
    user_id, user_data = next(iter(data.items()))
    print(f"User data fetched for ID: {user_id}")
    return user_data

user_data = fetch_user_data()
if not user_data:
    print("Error: No user data available for training.")
    exit(1)

# Extract relevant fields
name = user_data.get("name", "Avatar")
tweets = user_data.get("tweets", [])
personality_responses = [
    f"{key}: {value}"
    for key, value in user_data.items()
    if key != "name" and key != "tweets"
]

# Prepare dataset
output_file = "user_dataset.txt"
with open(output_file, "w", encoding="utf-8") as f:
    if tweets:
        f.write("\n".join(tweets) + "\n")
    if personality_responses:
        f.write("\n".join(personality_responses) + "\n")

print(f"Data saved to {output_file}")

# Load GPT-2 model and tokenizer
model_name = "gpt2"
print("Loading model and tokenizer...")
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Prepare dataset for fine-tuning
def load_and_tokenize_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # Tokenize the dataset
    encodings = tokenizer(lines, truncation=True, padding=True, max_length=128, return_tensors="pt")
    return Dataset.from_dict({"input_ids": encodings["input_ids"], "attention_mask": encodings["attention_mask"]})

dataset = load_and_tokenize_dataset(output_file)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Training arguments
training_args = TrainingArguments(
    output_dir=f"./models/{name.replace(' ', '_').lower()}_avatar",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

# Train the model
print("Training the model...")
trainer.train()

# Save the model
output_dir = f"./models/{name.replace(' ', '_').lower()}_avatar"
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model trained and saved to {output_dir}")
