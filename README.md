# The Hangman Challenge with Bidirectional LSTM RNN

This project implements a Hangman game using a Bidirectional LSTM RNN model to predict letters in an attempt to solve the word puzzle. The model is trained on a dictionary of words and is designed to guess the correct letters based on previously guessed letters, utilizing advanced natural language processing techniques.

## Table of Contents
- [Introduction](#introduction)
- [Inspiration](#inspiration)
- [Approach](#approach)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training and Results](#training-and-results)
- [Challenges Faced](#challenges-faced)
- [Future Improvements](#future-improvements)
- [How to Use](#how-to-use)
- [Contributing](#contributing)
- [License](#license)

## Introduction
The goal of this project is to create an intelligent agent for playing the classic Hangman game using machine learning techniques. The Bidirectional LSTM model is utilized to improve context understanding and pattern recognition in predicting letters, aiming to outperform traditional approaches in solving the word puzzle.

## Inspiration
This approach is motivated by the limitations of traditional LSTM and GRU models in handling long-range dependencies and recognizing complex patterns in text-based tasks. Bidirectional LSTM provides the ability to better understand both past and future contexts when predicting the next letter in the Hangman word sequence.

## Approach
- The model was built using a **Bidirectional LSTM** architecture, which allows it to process sequences from both directions, leading to improved pattern recognition capabilities.
- An **embedding layer** was used to convert input characters into dense vector representations, providing a more meaningful input to the LSTM layers.
- The **Dense layer** with a softmax activation function was used to output probabilities for each character, allowing the model to predict the most likely letter in the sequence.

## Data Preprocessing
- A character-to-index mapping was implemented to convert each letter in the training words into numerical sequences.
- Words were padded or truncated to a uniform length to ensure that all input sequences were of consistent size.
- The dataset was further processed to generate input-output pairs suitable for training the model, ensuring that the LSTM model could learn from varying word lengths.

## Training and Results
- The model was trained over **10 epochs**, during which both training and validation accuracy gradually improved.
- Initial results showed a training accuracy of approximately **63.67%** with a corresponding validation accuracy of **66.96%**, which improved to around **68.86%** training accuracy and **68.35%** validation accuracy by the end of training.
- Despite these improvements in accuracy, the practical success rate of the Hangman game remained low at **6.2%** after running multiple practice games, highlighting the challenge of achieving high performance in this task.

## Challenges Faced
- **Low Success Rate:** Even though the model accuracy showed promising numbers during training, translating this to a higher success rate in actual gameplay proved to be challenging.
- **Sequence Length Handling:** Managing input sequence lengths for varying word sizes and handling prediction errors were significant hurdles during model development.
- **Data Complexity:** The complexity of the Hangman game, which involves both strategic guessing and language patterns, posed a challenge in aligning model predictions with human-like decision-making.

## Future Improvements
- **Reinforcement Learning:** Implement techniques that allow the model to learn from its mistakes and adapt its guesses based on past attempts, aiming to improve its success rate.
- **Advanced NLP Models:** Experiment with more sophisticated models like Transformers, which could enhance context understanding and prediction accuracy.
- **Hyperparameter Optimization:** Further fine-tune the model's hyperparameters to optimize its performance and reduce the loss.

## How to Use
1. **Clone the repository:**
   ```bash
   git clone https://github.com/mathew-2/The_Hangman_Challenge.git
   cd The_Hangman_Challenge

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt

3. **Run the Python script to train the model or test the Hangman game:**
   ```bash
   python scripts/Hangman_api_user.py

## Contributing
Contributions are welcome! Feel free to submit issues or pull requests to help improve the project.
