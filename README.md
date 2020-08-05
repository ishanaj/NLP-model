# NLP-model

# Methodology Used
### Step I: Pre-processing
It includes tokenizing, removing stop words and punctuation, and POS tagging, combining two consequent noun occurrences
### Step II: Clustering the aspects terms and sentiments
This is done by creating a dependency tree using stanza (earlier Stanford NLP) and clustering aspects and sentiment determining terms according to their relation. Further, these are filtered for the aspect terms (only nouns and adjectives).
### Step III: Determining the aspect category from aspect terms
Word2vec pre-trained model by Google can be used to train on these aspect terms. Once the model is trained, the cosine similarity can be used to categorize the aspect terms into the given categories.
### Step IV: Determining the sentiment of the related aspect terms
Every aspect term is clustered with its dependent term, which can be used to determine the sentiment polarity. The sentiment polarity can be found using Sentiment Intensity Analyser from nltk library.
### Step V: Determining the target term
If two locations are mentioned in a sentence, the dependent term from the dependency tree gives us its related aspect. However, if the sentence mentions only one of them, it would be describing that particular location.
 
# Pros and Cons of the model prediction
The model extracts the aspect terms and clusters them accurately. Aspects like touristy, transit-location, shopping, green-nature, multi-cultural are determined with much more accuracy than the others.

The Word2Vec model will have to be trained more efficiently to determine the most similar aspect category to the extracted aspect terms.
The sentiments can be predicted in a better way if we use the dependency tree too to find any other sentiments related to it which either negates or amplifies the polarity of the sentiment.

For e.g., a sentence containing terms like “not very good”. The model would consider good and very to point towards positive polarity, however, if we go deep into the dependency tree, we might find a negation, which would invert the polarity.


# Short note on favourite Machine Learning library
I find KERAS to be a very powerful library. Keras overpowers other machine learning libraries, similar to how python dominates other languages in terms of ease of writing code and fast deployment of projects/models. One can simply transfer his/her thinking into a piece of code.

Keras has many pre-trained models like MobileNet, DenseNet, ImageNet, etc. which can be directly used for making predictions, feature learning, or transfer learning.
Apart from this, it also supports working on multiple backends like Tensorflow, Theano. We can train a model on Keras while testing it on some other backend. Keras has built-in support for working on multiple GPU’s, enabling data parallelism.

However apart from being such a powerful library, it of course is a bit slower than other libraries. Also, Keras does not provide rich support for data pre-processing. This is where libraries Scikit-learn outshine Keras.
 

