import json
import random
import torch
from model import NeuralNet
from nltk_utils import tokenize, bag_of_words

# Choose the appropriate device (GPU or CPU) based on availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the chatbot intents and pre-trained model data from files
with open('intents.json') as f:
    intents = json.load(f)

data = torch.load("data.pth")

# Instantiate the model using the loaded data and transfer it to the selected device
model = NeuralNet(data["input_size"], data["hidden_size"], data["output_size"]).to(device)
model.load_state_dict(data["model_state"])
model.eval()  # Set the model to evaluation mode

bot_name = "Sam"
print("Let's chat! (type 'quit' to exit)")

while True:
    sentence = input("You: ")
    # Allow the user to exit the chat by typing 'quit', case-insensitively
    if sentence.lower() == "quit":
        break

    # Convert the user input into a format suitable for model prediction
    sentence_tokens = tokenize(sentence)
    X = bag_of_words(sentence_tokens, data['all_words']).reshape(1, -1)
    X = torch.from_numpy(X).float().to(device)  # Convert to a torch tensor and send to device

    # Make a prediction based on the user input and evaluate the confidence
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = data['tags'][predicted.item()]
    prob = torch.softmax(output, dim=1)[0][predicted.item()]

    # Respond to the user based on the model's prediction confidence
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
                break  # Stop searching once a matching intent is found
    else:
        print(f"{bot_name}: I do not understand...")
