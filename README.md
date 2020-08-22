# Sentiment Analysis with Deep Learning, using BERT
Sentiment Analysis with Deep Learning of Twitter Smile emotion dataset, using BERT

#### Objective:

Perform Sentiment Analysis, to classify the emotion of tweets w.r.t. British museums, using pretrained BERT model.

#### What is BERT?

BERT is a highly used Machine Learning model in NLP(Natural Language Processing) subspace. It is a large scale model that has transformers inside it. It's an advancement of RNN (Recurrent Neural Network) i.e., It's able to parallelize processing and training at inference (similar to CNN, where, input size is fixed). As earlier in RNN, we might have to process each word independently and now, we can do things in parallel. And, that's the real goal. BERT is trained on unsupervised data, i.e., data with no labels, from a huge corpus of data present over Intenet.

![alt text](https://github.com/rickhagwal/Bert_Sentiment_Analysis/blob/master/images/BERT_diagrams.png)

#### Dataset:

Wang, Bo; Tsakalidis, Adam; Liakata, Maria; Zubiaga, Arkaitz; Procter, Rob; Jensen, Eric (2016): [SMILE Twitter Emotion dataset] (https://doi.org/10.6084/m9.figshare.3187909.v2)
It contains collection of tweets of British Museums and each tweet has been labelled, based on the emotion inside that tweet.

![alt text](https://github.com/rickhagwal/Bert_Sentiment_Analysis/blob/master/images/actual_labels.PNG)

#### Technologies and Libraries used:
HuggingFace(It's a company that has developed Transformers, whose methods can be used within PyTorch, TensoFlow 2.0), 
PyTorch,
BERT,
Pandas,
NumPy

#### Project Outline:

Task 1: Exploratory Data Analysis and Preprocessing

Task 2: Training/Validation Split

Task 3: Loading Tokenizer and Encoding our Data

Task 4: Setting up BERT Pretrained Model

Task 5: Creating Data Loaders

Task 6: Setting Up Optimizer and Scheduler

Task 7: Defining our Performance Metrics

Task 8: Creating Training Loop

Task 9: Loading and Evaluating our Model


#### Model Training:

Loaded in pre-trained BERT model, encode(To Convert text into numerical data) it with custom Output layer. And then, split datset into train and validation datsets, using stratified approach, to preserve classes in same ratio in both training and test set. Here used, test set as 15%. Thus, for each class, 85% data should be in train dataset and 15% data should be in valid dataset.

#### Performance Metrics used:

Used F-1 score metrics, since their is severe class imbalance in the dataset. For that, we used stratified approach while splitting dataset into train and validation.

![alt text](https://github.com/rickhagwal/Bert_Sentiment_Analysis/blob/master/images/after%20stratify.PNG)

#### Results:

Results are pretty impressive , as can be seen below-

![alt text](https://github.com/rickhagwal/Bert_Sentiment_Analysis/blob/master/images/Output.PNG)

#### BERT Scope and Problems:

BERT can be adapted to do question-answering, multiple choice, sentence completion and predicting missing words.

Problems are:

1.) It's Speed. BERT can be slow for production.

2.) It might have biases, as data it's trained on is from Internet i.e., Internet data might more provide data that represnts wealthier countries more than the poorer ones. e.g., data might be more available from countries like- U.S., U.K. compared to South Africa or Zimbabwe.

