
# Consumer Reviews Analysis Project

The dataset is from Kaggle:
And it can be downloaded in this link: https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products/data
In Google Drive: https://drive.google.com/drive/folders/1oVaRhOykQpQlN0-MB4ga8Y-mGlsOKNEs?usp=sharing

## Overview

This project aims to analyze and classify consumer reviews using natural language processing (NLP) techniques. The analysis includes sentiment classification, category grouping, and generating text summaries of the best and worst products in a given category based on user reviews. The notebook leverages transformers for sentiment analysis and text generation, and it is designed to help potential shoppers make better purchasing decisions.

The notebook focuses on:
- Grouping products into broader categories.
- Classifying sentiment (positive, neutral, negative) for each review.
- Generating brief summaries or articles highlighting the best and worst products within a category.

## Notebook Structure

The notebook is organized into the following sections:

1. **Data Loading & Preprocessing**:
   - Loads the dataset(s) with product reviews.
   - Cleans the data, focusing on important columns such as `name`, `categories`, and `reviews.text`.
   - Maps product categories into broader categories to simplify analysis.

2. **Sentiment Analysis**:
   - Applies a sentiment analysis model to categorize reviews as positive, neutral, or negative.
   - Sentiment classification is key to determining the quality of products within each category.

3. **Category Grouping**:
   - Groups product categories into new, simplified categories for easier understanding and processing.
   - Uses transformer models like MiniLM-L6-v2 for fast and efficient category mapping.

4. **Best and Worst Product Identification**:
   - Uses review ratings to select the 3 best and 3 worst products in each category.
   - Sentiment analysis is leveraged to assist in determining product quality.

5. **Text Generation**:
   - Implements an LLM-based text generator that creates 250-word summaries about the best or worst products in each category.
   - Generates product summaries by synthesizing review sentiments and product details.

## Requirements

Before running the notebook, make sure you have the following dependencies installed:

- Python 3.x
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- Transformers (Hugging Face)
- TensorFlow or PyTorch (for transformer models)
- Gradio (for deployment if needed)

You can install the required packages using:

```bash
pip install pandas numpy scikit-learn transformers torch gradio
```

## How to Run

1. **Download the Dataset**:
   - Place your product review dataset in the same directory as the notebook.
   - The dataset should contain product reviews, categories, and ratings.

2. **Run the Notebook**:
   - Open the notebook in JupyterLab or Jupyter Notebook.
   - Execute the cells step by step to load data, preprocess, and perform analysis.
   - The sentiment classification and text generation will run automatically on the dataset.

3. **Evaluate Results**:
   - The notebook will produce sentiment-labeled reviews and generate summaries of the best and worst products in each category.
   - You can modify the prompt to generate specific product descriptions as needed.

## Future Improvements

- Implement additional models for category mapping to improve accuracy.
- Experiment with LoRA and other prompt engineering techniques to enhance text generation.
- Expand the dataset to include a larger variety of product categories and review samples.

## Contributing

If you would like to contribute to this project, feel free to fork the repository and submit a pull request. Please include tests with your contributions to ensure quality and stability.

## License

This project is licensed under the MIT License.
