# GRU-Based Sentiment Analysis

This project explores sentiment analysis on text data using Gated Recurrent Units (GRU), including custom GRU implementations and variations utilizing PyTorch's native GRU and bidirectional GRU layers. The project aims to classify text data into positive or negative sentiments, showcasing the effectiveness of GRU models in handling sequence data and capturing temporal dependencies.

## Project Structure

- `CustomGRU.py`: Contains the implementation of a custom GRU model.
- `PyTorchGRU.py`: Utilizes PyTorch's native GRU for sentiment analysis.
- `BidirectionalGRU.py`: Explores the use of a bidirectional GRU model for improved context understanding.
- `data/`: Directory containing the sentiment dataset split into training and validation sets.
- `word2vec/`: Contains pre-trained Word2Vec embeddings for text representation.

## Getting Started

### Dependencies

Ensure you have the following installed:
- Python 3.8 or above
- PyTorch 1.7.1 or above
- Gensim for Word2Vec embeddings
- Matplotlib and Seaborn for visualization
- Scikit-learn for evaluation metrics

### Dataset

The dataset comprises positive and negative reviews, preprocessed and vectorized using Word2Vec embeddings. Training and validation sets are located in the `data/` directory.

### Training the Models

To train the models, navigate to the respective script and run:

```bash
python CustomGRU.py
python PyTorchGRU.py
python BidirectionalGRU.py