# Chatbot with PyTorch
This project presents a straightforward implementation of a contextual chatbot using PyTorch, designed to be accessible for beginners. The chatbot is built on a feed-forward neural network with two hidden layers, demonstrating the core principles of chatbot development.

Key Features:
- Simplicity: The codebase is kept simple and readable to help beginners understand the mechanics of chatbots.
- Customizability: Users can easily adapt the chatbot for various use cases by modifying the intents.json file, which defines the bot's responses and patterns.
- Feed-Forward Neural Network: Employs a neural network architecture with two hidden layers to process and respond to user inputs.

Getting Started:

Prerequisites
- Python 3.x
- PyTorch
- NLTK

Installation:

1. Clone the repository and create a virtual environment:
- git clone <repository-url>
- cd <repository-directory>
- python3 -m venv venv

2. Activate the virtual environment:
- On Mac/Linux: . venv/bin/activate
- On Windows: venv\Scripts\activate

3. Install the required packages:
- pip install torch nltk

4. Download NLTK tokenizers:
- python -c "import nltk; nltk.download('punkt')"

Training the Chatbot:
Run python train.py to start the training process. This script reads the intents.json file, trains the neural network model, and saves it to data.pth.

Chatting with the Chatbot:
After training, run python chat.py to start a chat session with the chatbot. The script loads the trained model and interacts with the user in the console.

Customization:
To customize the chatbot:

1. Edit the intents.json file to include your desired conversation patterns and responses.
2. Re-train the chatbot by running python train.py.
