import streamlit as st
import pandas as pd
import os
import numpy as np
import pickle
import re
import matplotlib.pyplot as plt
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
# from ipynb.fs.full.Sentiment_Analysis import review_extraction # takes
# too much time so decided to just rewrite the function
import plotly.express as px


st.title('IMDB reviews Sentiment Analysis')
st.markdown('##### We are using a total of 25K reviews to train the \
model, 12.5K positive and 12.5K of each kind, feel free to go to the \
side bar to get a sample of the comments by checking the box. ')
# side work because importing it from ipynb file and loading it to streamlit
# take too much time, we are also only loading 100 comments from the data.
# to make the loading faster
positive_dir = './train/pos/'
negative_dir = './train/neg/'
@st.cache(allow_output_mutation=True,persist = True)
def review_extraction(directory,max = 100):
    '''
    extracting reviews from a directory
    '''
    reviews = []
    filenames = os.listdir(directory)
    count = 0
    for filename in filenames:
        count+=1
        with open(directory + filename,'r') as f:
            reviews.append(f.read() )
            if count == max:
                break
    return reviews
positive_reviews = review_extraction(positive_dir)
negative_reviews = review_extraction(negative_dir)

reviews0 = pd.DataFrame(np.c_[negative_reviews,np.zeros(len(negative_reviews),dtype = 'uint8')],
                        columns = ['reviews','sentiment'])
reviews1 = pd.DataFrame(np.c_[positive_reviews,np.ones(len(negative_reviews),dtype = 'uint8')],
                        columns = ['reviews','sentiment'])
## end of side work

# showing radom positive / negative comments
st.sidebar.subheader('Displaying comments')
if not st.sidebar.checkbox('Hide comments display', True):
    st.subheader('Comments Display')
    select = st.radio('Comment type',('positive','negative'))
    if select == 'positive':
        st.markdown(reviews1[reviews1['sentiment'] == '1']['reviews'].sample(1).iloc[0])
    else :
        st.markdown(reviews0[reviews0['sentiment'] == '0']['reviews'].sample(1).iloc[0])

##

# side work, getting functions from notebook because loading them takes more time for streamlit
porter = PorterStemmer()
def preprocessor(txt):
    '''
    light preprocessing of the text
    '''
    txt+= ' ' # adding space for the emogies not to stick to last word
    # removing html tags
    txt = re.sub('<[^>]*>',' ', txt)

    # puting emoticons to the end and puting everything to lowercase
    em_reg = '(\:\w+\:|\<[\/\\]?3|[\(\)\\\D|\*\$][\-\^]?[\:\;\=]|[\:\;\=B8][\-\^]?[3DOPp\@\$\*\\\)\(\/\|])(?=\s|[\!\.\?]|$)'
    emoticons = re.findall(em_reg,txt)
    txt = re.sub(em_reg ,' ', txt.lower() ) + ' '.join(emoticons)
    return txt
def tokenizer_porter(text,stop = None):
    if type(stop) == type(None):
        return [porter.stem(word) for word in text.split()]
    else :
        return [porter.stem(word) for word in text.split() if word not in stop]
stop = stopwords.words('english')
##

# loading model
filename = 'LR_model.sav'
clf = pickle.load(open(filename,'rb'))
tfidf = pickle.load(open('tfidf.sav','rb'))
#

# wordcloud
negative_words = 'negative '
positive_words = 'positive '
st.sidebar.subheader("Write your own words and see where it's classified in the Word Cloud")
if not st.sidebar.checkbox('Hide Word cloud',True):
    word = st.text_input('Your word','write words separated with space')
    if word != 'write words separated with comma':
        word = word.split()
        senti = np.array(clf.predict(tfidf.transform(word)))
        positive_words += ' '.join(np.array(word)[senti ==1])

        wordcloud0 = WordCloud(stopwords = STOPWORDS, background_color = 'white',\
        height = 650,width = 800).generate(positive_words)
        plt.subplot(1,2,1)
        plt.title('Positive')
        plt.imshow(wordcloud0)
        plt.axis('off')

        negative_words += ' '.join(np.array(word)[senti ==0])
        wordcloud1 = WordCloud(stopwords = STOPWORDS, background_color = 'black',\
        height = 650,width = 800).generate(negative_words)
        plt.subplot(1,2,2)
        plt.title('Negative')
        plt.imshow(wordcloud1)
        plt.axis('off')
        st.pyplot()


##

# writing own review
st.markdown('## Write down your own review and let the model predict its Sentiment : ')
review = st.text_input('Your Review','review   ')
if review != 'review   ' and review != '':
    review_pro = tokenizer_porter(preprocessor(review),stop)
    review_vec = tfidf.transform(review_pro)
    pred = np.mean(clf.predict(review_vec))
    review_pro = ' '.join(tokenizer_porter(preprocessor(review),stop))
    review_vec = tfidf.transform([review_pro])
    prediction = clf.predict(review_vec)[0] + pred


    if prediction < 0.5:
        st.write(" That's a bad review, it will probably have less than 4 stars.")
    elif prediction > 0.5:
        st.write(" That's a good review, it will probably have more than 7 stars.")
    else:
        st.write(" That's a neutral review, it will probably between 4 and 7 stars.")
##

st.markdown('###### this is just a small vizualization, you can find the whole work \
in the jupyter notebook')
