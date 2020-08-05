import json
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer 
import stanfordnlp
import stanza 
from gensim.models import Word2Vec
from numpy import dot
from numpy.linalg import norm
import copy
from nltk.sentiment.vader import SentimentIntensityAnalyzer



### download requirements
# stanfordnlp.download('en')
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

""" extracting data from the json file"""
f = open("sentihood-test.json",)

data = json.load(f)

list_of_sentences = []
list_of_aspects = []
list_of_sentiments = []
feedback_data = []
for feedback in data:
    feedback_data.append(feedback)
    aspects = []
    sentiments = []
    if feedback["opinions"]:
        list_of_sentences.append(feedback["text"].lower())
        for opinions in feedback["opinions"]:
            aspects.append(opinions["aspect"])
            sentiments.append(opinions["sentiment"])
        list_of_aspects.append(aspects)
        list_of_sentiments.append(sentiments)


""" tokenizing and POS tagging each sentence """
tagged_sentences = []
t_list = []
for sentence in list_of_sentences:
    sentence.lower()
    tokenized_sentence = nltk.word_tokenize(sentence)
    tagged_sentences.append(nltk.pos_tag(tokenized_sentence))
    l = [   ]
    for word in tokenized_sentence:
        if word in ["location1", "location2"]:
            l.append(word)
    t_list.append(l)

""" joining two probale aspect terms (nouns), which occur one after
the other and forming new sentences. For e.g night life ----> nightlife. """
new_list_of_sentences = []
new_list_of_words = []
for sentence in tagged_sentences:
    f = 0
    new_sentence = []
    for i in range(len(sentence)-1):
        if sentence[i][1] in ["NN", "NNS"] and sentence[i+1][1] in ["NN", "NNS"]:
            new_sentence.append(sentence[i][0] + sentence[i+1][0])
            f = 1
        else:
            if f:
                f = 0
            else:
                new_sentence.append(sentence[i][0])
                if i == len(sentence)-2:
                    new_sentence.append(sentence[i+1][0])
    new_list_of_words.append(new_sentence)
    new_list_of_sentences.append(" ".join(new_sentence))


stop_words = set(stopwords.words('english'))

""" POS tagging the new sentences formed"""
tagged_sentences = []
for sentence in new_list_of_sentences:
    tokenized_sentence = nltk.word_tokenize(sentence)
    filtered_sentence = [word for word in tokenized_sentence if not word in stop_words]
    tagged_sentences.append(nltk.pos_tag(filtered_sentence))

""" using stanza (earlier called stanfordnlp library) to form a dependency tree """
nlp = stanza.Pipeline()
list_of_nodes = []

for sentence in new_list_of_sentences:
    doc = nlp(sentence)
    node = []
    for edge in doc.sentences[0].dependencies:
        node.append([edge[2].text, edge[0].text, edge[1]])

    list_of_nodes.append(node)

""" listing the words than can be the aspect and sentiment terms (nouns, adjectives)"""
aspect_list = []
aspect_category = []
for sentence in tagged_sentences:
    aspect = []
    category = []
    for x in sentence:
        if x[1]=='JJ' or x[1]=='NN' or x[1]=='JJR' or x[1]=='NNS' or x[1]=='RB':
            aspect.append(list(x))
            category.append(x[0])
    aspect_list.append(aspect)
    aspect_category.append(category)



""" making a cluster of the aspects and the sentiment according 
 to the dependency relation """
aspect_cluster = []
for pos in range(len(list_of_nodes)):
    aspects = aspect_list[pos]
    nodes = list_of_nodes[pos]

    cluster = []
    for i in aspects:
        # print(i)
        l = []
        for j in nodes:
            if((j[0]==i[0] or j[1]==i[0]) and (j[2] in ["nsubj", "acl:relcl", "obj", "dobj", "agent", "advmod", "amod", "neg", "prep_of", "acomp", "xcomp", "compound"])):
                if j[0] == i[0]:
                    l.append(j[1])
                else:
                    l.append(j[0])
        cluster.append([i[0], l])
    aspect_cluster.append(cluster)


""" filtering the cluster"""
filtered_cluster = []
d = {}
for pos in range(len(list_of_nodes)):
    aspects = aspect_list[pos]
    for i in aspects:
        d[i[0]] = i[1]

for cluster in aspect_cluster:
    fcluster = []
    for i in cluster:
        if d[i[0]] in ["NN", "NNS", "NNP", "NNPS"]:
            fcluster.append(i)
    filtered_cluster.append(fcluster)


""" extracting aspect terms from the filtered cluster to be put into Word2vec for training"""
extracted_aspect_terms = []

for i in range(len(filtered_cluster)):
    sentence = filtered_cluster[i]
    aspect = list_of_aspects[i]
    aspect_terms = []
    for cluster in sentence:
        if not cluster[0] in ["location1", "location2", "location"]:
            aspect_terms.append(cluster[0])
    extracted_aspect_terms.append(aspect_terms)

temp_extracted_aspect_terms = copy.deepcopy(extracted_aspect_terms)

aspect_words = ["live", "safety", "price", "quiet", "dining","nightlife", "transit-location", "touristy", "shopping", "green-nature",  "multicultural", "general"]

""" adding aspect words in the list"""
for x in aspect_words:
    temp_extracted_aspect_terms.append([x])

"""" training a word2vec model with our list of words"""
model = Word2Vec(temp_extracted_aspect_terms, min_count=1,size= 50,workers=3, window =3, sg = 1) 

""" function to calculate cosine simmilarity between a word and all aspect categories
 generates a sorted dictionary of similarity with all the aspect categories"""
def cosine_distance(model, word,target_list) :
    cosine_dict ={}
    word_list = []
    a = model[word]
    for item in target_list :
        # if item == word :
        b = model [item]
        cos_sim = dot(a, b)/(norm(a)*norm(b))
        cosine_dict[item] = [cos_sim]
    dist_sort=sorted(cosine_dict.items(), key=lambda dist: dist[1],reverse = True) ## in Descedning order 
    return dict(dist_sort)

""" Sentiment Intensity analyser for determining the polarity of the aspect"""
sid = SentimentIntensityAnalyzer()

""" determining the result"""
result = []
for i in range(len(filtered_cluster)):
    sentence = filtered_cluster[i]
    l = []
    feedback = feedback_data[i]
    feedback["model_pred"] = []
    target = None
    prev_aspects = []
    if len(t_list[i]) == 1:
            target = t_list[i][0]
    for term in sentence:
        d = {}
        if term[0] in ["location1", "location2", "location"]:
            target = term[0]
        else:
            d["target_entity"] = target

            aspect = list(cosine_distance(model, term[0], aspect_words).keys())[0]
            if aspect not in [prev_aspects]:
                d["aspect"] = aspect
                polarity = 0
                for word in term[1]:
                    if (sid.polarity_scores(word)['compound']) >= 0:
                        polarity *= 1
                    else:
                        polarity *= -1
                prev_aspects.append(aspect)
                d["sentiment"] = "Positive" if polarity == 1 else "Negetive"
                feedback["model_pred"].append(d)
    result.append(feedback)

with open("result.json", "w") as outfile:
    json.dump(result, outfile, indent=4) 




