
---

# **Fake News Detection Using Transformer-Based Models (BERT & CT-BERT)**

This project focuses on **fake news detection** using **Transformer-based models** like **BERT** and **CT-BERT**. The model classifies **tweets** as **real** or **fake**, leveraging **pre-trained language models** and **fine-tuning** techniques.

---

## **Table of Contents**
- [Project Overview](#project-overview)
- [Dataset & Visualization](#dataset--visualization)
- [Installation](#installation)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Training & Evaluation](#training--evaluation)
- [Results & Analysis](#results--analysis)
- [Future Work](#future-work)
- [References](#references)

---

## **Project Overview**
- **Objective:** Classify tweets as **real or fake** using **Transformer-based NLP models**.
- **Models Used:**
  - **BERT** (Google's pre-trained model)
  - **CT-BERT** (COVID-specific fine-tuned BERT)
- **Approaches:**
  - **Fine-tuning BERT and CT-BERT**
  - **Feature-based learning with BiGRU**
  - **Evaluation using accuracy, precision, recall, and F1-score**

---

## **Dataset & Visualization**
The dataset consists of **COVID-19 related tweets** labeled as **real or fake**:
- **Training Set:** `Constraint_Train.csv`
- **Validation Set:** `Constraint_Val.csv`
- **Test Set:** `english_test_with_labels.csv`

### **Class Distribution**
The dataset has an **imbalanced distribution**, with more real news samples than fake ones. Below is a visualization:

```python
plt.figure(figsize=[7,5])
plt.bar(class_name, class_C, color='green', edgecolor='black')
plt.title(f'Class Distribution in Training Data \n Total Count of Samples: {sum(class_C)}')
plt.show()
```

![Class Distribution](results/class_distribution.jpg)

### **Word Cloud Visualization**
The most frequent words in **real** and **fake** tweets are visualized using **word clouds**.

```python
wordcloud(str(np.array(pre_Data)[np.where(np.array(Target) == 0)[0]]), 'Fake News')
wordcloud(str(np.array(pre_Data)[np.where(np.array(Target) == 1)[0]]), 'Real News')
```

#### **Fake News Word Cloud**
![Fake News Word Cloud](results/fake_news_wordcloud.jpg)

#### **Real News Word Cloud**
![Real News Word Cloud](results/real_news_wordcloud.jpg)

---

## **Installation**
Before running the code, install the required dependencies:

```bash
pip install torch torchvision transformers numpy pandas nltk matplotlib emoji wordcloud scikit-learn
```

If using **Google Colab**, install additional packages:

```bash
pip install datasets
```

---

## **Preprocessing**
The text is preprocessed using **tokenization, lemmatization, and removing stop words**.

```python
def pre_prosse(x):
    lemmatizer = WordNetLemmatizer()
    x = x.lower()
    x = emoji.demojize(x)
    x = re.sub(r'http\S+|www\S+', '', x)
    x = re.sub(r'[^\w\s]', ' ', x)
    x = re.sub(r'\d+', '', x)
    x = word_tokenize(x)
    x = [token for token in x if token not in stop_words]
    return [lemmatizer.lemmatize(i) for i in x]
```

Then, the dataset is tokenized using **BERT Tokenizer**:

```python
def tokenizer_F(X):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer(X, padding=True, truncation=True, return_tensors="pt", max_length=128)
    return inputs["input_ids"], inputs["attention_mask"]
```

---

## **Model Architecture**
This project implements **four different models**:

1. **BERT with a Fully Connected (Dense) Layer (Frozen BERT)**
   - Uses **pre-trained BERT** as a feature extractor.
   - A **dense output layer** classifies the news.

2. **BERT with BiGRU (Frozen BERT)**
   - Uses **BERT embeddings**.
   - A **BiGRU (Bidirectional Gated Recurrent Unit)** processes sequences.
   - A **dense output layer** classifies the news.

3. **BERT with a Fully Connected (Dense) Layer (Fine-Tuned BERT)**
   - BERT is **fine-tuned** during training.
   - A **dense output layer** classifies the news.

4. **BERT with BiGRU (Fine-Tuned BERT)**
   - The entire **BERT model is trainable**.
   - BiGRU processes sequences before classification.

---

## **Training & Evaluation**
The models are trained and evaluated using:
- **Binary Cross-Entropy Loss**
- **Adam Optimizer**
- **Accuracy, Precision, Recall, and F1-score**

### **Training Process**
```python
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

value = train_and_evaluate(model, criterion, optimizer, dataloader_train, dataloader_Val, num_epochs, device)
plot(num_epochs, value)
```

### **Testing Process**
```python
true_labels, predicted_labels = test_F(model, dataloader_Test, class_name)
```

---

## **Results & Analysis**
The models were evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**

### **Loss Function Plot**
<table>
    <tr>
        <td><img src="results/loss_plot1.jpg" width="300"/></td>
        <td><img src="results/loss_plot2.jpg" width="300"/></td>
    </tr>
</table>

### **Confusion Matrix**
<table>
    <tr>
        <td><img src="results/confusion_matrix1.jpg" width="300"/></td>
        <td><img src="results/confusion_matrix2.jpg" width="300"/></td>
    </tr>
</table>

### **Misclassified Tweets**
```python
def print_wrong_prediction(true_labels, predicted_labels, Data_Test):
    print('Incorrectly classified tweets: \n')
    for i in range(true_labels.size):
        if true_labels[i] != predicted_labels[i]:
            print(f'Tweet {i}: {Data_Test[i]}')
```

Example Output:
```
Tweet 42: "Breaking news! Covid-19 is completely cured. Scientists confirm!"
Predicted: Real
Actual: Fake
```

---

## **Future Work**
- **Experiment with RoBERTa and XLNet**.
- **Use larger datasets** for better generalization.
- **Incorporate adversarial training** for robustness.

---

## **References**
- **BERT Paper**: *Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"*  
- **CT-BERT Paper**: *Müller et al., "COVID-Twitter-BERT: A Natural Language Processing Model to Analyze COVID-19 Content on Twitter"*  
- **Hugging Face Transformers Library**: https://huggingface.co/docs/transformers/  

---
