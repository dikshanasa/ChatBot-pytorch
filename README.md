# ChatBot using AIML
Chatbots are software applications that use artificial intelligence & natural language processing to understand what a human wants, and guides them to their desired outcome with as little work for the end user as possible. Like a virtual assistant for your customer experience touchpoints.
In this project we have built a Chatbot using PyTorch and focussing on the use of Neural Networks.
## Requirements 
<li>Python 3
<li>Dictionaries & Lists
<li>Numpy
<li>Pandas
<li>Pytorch
<li>Natural Language Processing (Bag of Words)
</br>

### Software Used - Google Colab
# Building our Chatbot 
## Getting Ready
### Importing Libraries
```
import numpy as np
import random
import json
import nltk
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
```
### Create our functions 
```
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())
```
### Stemming
If we have 3 words like “walk”, “walked”, “walking”, these might seem different words but they generally have the same meaning and also have the same base form; “walk”. So, in order for our model to understand all different form of the same words we need to train our model with that form. This is called Stemming. There are different methods that we can use for stemming. Here we will use Porter Stemmer model form our NLTK Library.
```
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
```
### Bag of Words
We will be splitting each word in the sentences and adding it to an array. We will be using bag of words. Which will initially be a list of zeros with the size equal to the length of the all words array.
```
def bag_of_words(tokenized_sentence, words):
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag
```
## Loading the Data & Cleaning it
We will be using a data set called intents.json which has the following structure.
```
{"intents": [
        {"tag": "greeting",
         "patterns": ["Hi there", "How are you", "Is anyone there?","Hey","Hola", "Hello", "Good day"],
         "responses": ["Hello", "Good to see you again", "Hi there, how can I help?"],
         "context": [""]
        },
        {"tag": "goodbye",
         "patterns": ["Bye", "See you later", "Goodbye", "Nice chatting to you, bye", "Till next time"],
         "responses": ["See you!", "Have a nice day", "Bye! Come back again soon."],
         "context": [""]
        },
        {"tag": "thanks",
         "patterns": ["Thanks", "Thank you", "That's helpful", "Awesome, thanks", "Thanks for helping me"],
         "responses": ["Happy to help!", "Any time!", "My pleasure"],
         "context": [""]
        },
        {"tag": "noanswer",
         "patterns": [],
         "responses": ["Sorry, can't understand you", "Please give me more info", "Not sure I understand"],
         "context": [""]
        },
        {"tag": "options",
         "patterns": ["How you could help me?", "What help you provide?", "How you can be helpful?"],
         "responses": ["I can guide you through Adverse management problems, order tracking, person to be contacted and Department related queries","I can provide support related to following problems technical query,management related query,order related query,tracking related query,procurement query,outsourcing problem,manufacturing delay,"],
         "context": [""]
        }, {"tag":"domain",
        "patterns":["How to improve team members domain knowledge","improving domain knowledge of team members"],
        "responses":["set up key meetings and workshop,create a shared drive for information,Hold informal sharing session"],
        "context":[""]
    }
   ]
}
```
Now we will unpack our JSON with the following code :
```
all_words = []
tags = []
xy = []
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))
```
## Implementing custom functions
```
# stem and lower each word

ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]

# remove duplicates and sort

all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)
```
## Creating Training Dataset
We will transform the data into a format that our PyTorch Model can Easily Understand
```
# create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)
```
## PyTorch Model
---
Here we will be making a class to implement our custom neural network. It will be a feed forward neural Network which will have 3 Linear Layers and we will be using activation function “ReLU” . 

    Note: 
    We have used the super() function to inherit the properties of its parent class. 
    This is an Object Oriented Programming (OOP) concept.

### Feed Forward Neural Network

`A feedforward neural network is an artificial neural network wherein connections between the nodes do not form a cycle. As such, it is different from its descendant: recurrent neural networks.`


The feedforward neural network was the first and simplest type of artificial neural network devised In this network, the information moves in only one direction—forward—from the input nodes, through the hidden nodes (if any) and to the output nodes. There are no cycles or loops in the network [1] 


#### Activation Function:
`
An activation function is a function used in artificial neural networks which outputs a small value for small inputs, and a larger value if its inputs exceed a threshold. If the inputs are large enough, the activation function "fires", otherwise it does nothing. In other words, an activation function is like a gate that checks that an incoming value is greater than a critical number. [2]`


Activation functions are useful because they add non-linearities into neural networks, allowing the neural networks to learn powerful operations. If the activation functions were to be removed from a feedforward neural network, the entire network could be re-factored to a simple linear operation or matrix transformation on its input, and it would no longer be capable of performing complex tasks such as image recognition.


### ReLU Function:
There are a number of widely used activation functions in deep learning today. One of the simplest is the rectified linear unit, or ReLU function, which is a piece wise linear function that outputs zero if its input is negative, and directly outputs the input otherwise:

`f(x) = max(0,x) `
 
Mathematical definition of the ReLU Function

![relu](https://user-images.githubusercontent.com/64835443/163348174-039ef586-b5d9-422c-a48e-f4b6df184dc8.png)

 
Graph of the ReLU function, showing its flat gradient for negative x. 

#### ReLU Function Derivative:

It is also instructive to calculate the gradient of the ReLU function, which is mathematically undefined at x = 0 but which is still extremely useful in neural networks.

![reluderivative](https://user-images.githubusercontent.com/64835443/163348293-c346dce7-3c46-4ab1-ae8d-9f8ce4b07163.png)

The derivative of the ReLU function. In practice the derivative at x = 0 can be set to either 0 or 1. 
The zero derivative for negative x can give rise to problems when training a neural network, since a neuron can become 'trapped' in the zero region and backpropagation will never change its weights. [2] 
 
## Creating our Model
```
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out
 ```
 ### Assigning dataset to model 
 We will use some Magic functions, write our class.
 ```
 class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
  ```
  ## Using our Chatbot
  Our Model is Ready. Now we will finally chat with our chat-bot. As our training data was very limited, we can only chat about a handful of topics. You can train it on a bigger dataset to increase the chatbot’s generalization / knowledge.
  ```
  bot_name = "Sam"
print("Let's chat! (type 'quit' to exit)")
while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")
  ```
  ## Let's Chat with our Chatbot
  ![Chatbot_execution](https://user-images.githubusercontent.com/64835443/163354142-b917e1d3-ecc9-4ced-a7aa-1ab8f12cb4cd.gif)

