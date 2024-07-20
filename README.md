---

# ScribeSense: Student Responses Evaluation Platform

## Project Overview

ScribeSense is an innovative platform designed to automate the evaluation of student responses. By leveraging Natural Language Processing (NLP) techniques and a Large Language Model (LLM), ScribeSense provides objective and consistent grading, along with personalized feedback. This project specifically focuses on evaluating responses for CBSE 10th grade Physics, Chemistry, and English subjects.

## Features

- **Automated Grading**: Evaluates student responses by comparing them to vectorized answer representations using semantic similarity.
- **Personalized Feedback**: Generates tailored feedback highlighting areas for improvement and suggesting alternative phrasings.
- **NLP Techniques**: Utilizes text cleaning, tokenization, stop word removal, and lemmatization for accurate preprocessing.
- **Cosine Similarity**: Measures the semantic similarity between student responses and key answers.
- **Subject Classification (Future Work)**: Plans to include a module for automatic subject classification based on filenames.
- **Performance Analysis (Future Work)**: Incorporates time series analysis to track and improve student performance over time.

## Usage

1. **Upload Responses**: Students upload their .docx files containing their responses.
2. **Evaluation**: ScribeSense evaluates the responses and provides a score based on semantic similarity.
3. **Feedback**: Personalized feedback is generated for each response, highlighting areas for improvement and alternative phrasings.

## Data Set

ScribeSense utilizes a dataset specifically tailored for CBSE 10th grade Physics, Chemistry, and English. This dataset consists of hundreds of questions and expert-created answers, vectorized using word embedding techniques.

## Model

ScribeSense's model preprocesses student responses, compares them to vectorized answers using cosine similarity, and generates personalized feedback with the help of an LLM.

## Experiments

The model was tested using a sample assignment document, "Assignment.docx", containing five questions with varying levels of accuracy in student responses. The evaluation demonstrated ScribeSense's ability to provide accurate scoring and constructive feedback.

## Results

The results from the sample assignment indicated that ScribeSense can effectively assess student responses and offer meaningful feedback, showcasing its potential to streamline grading processes.

## Future Work

- **Automatic Subject Classification**: Enhance the model to automatically classify the subject of assignments based on filenames.
- **Time Series Analysis**: Implement techniques like SARIMA to analyze trends in student performance and provide targeted interventions.

## References

- [Text Vectorization](https://www.analyticsvidhya.com/blog/2021/06/part-5-step-by-step-guide-to-master-nlp-text-vectorization-approaches/)
- [Word Embedding](https://www.geeksforgeeks.org/python-word-embedding-using-word2vec/)
- [Lemmatization](https://www.analyticsvidhya.com/blog/2022/06/stemming-vs-lemmatization-in-nlp-must-know-differences/)
- [Sequential Analysis](https://medium.com/@abelkrw/a-guide-to-sequential-data-analysis-in-python-dcb6c929b7d6)
- [Ollama](https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings)

## Screenshots

### Homepage
![image](https://github.com/user-attachments/assets/2f00f315-598a-426e-840e-eafa8426c499)

### Student Document Page
![image](https://github.com/user-attachments/assets/b761d38c-60eb-4e32-85e5-ac4277cad114)

### Summary Sheet
![image](https://github.com/user-attachments/assets/d2fc88b0-d215-4564-a0cd-c6b19325571d)

---
