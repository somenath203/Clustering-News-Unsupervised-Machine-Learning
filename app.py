from fastapi import FastAPI
from pydantic import BaseModel
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import pickle


app = FastAPI()


model_kmeans = pickle.load(open('modelkmeans.pkl', 'rb'))
model_hierarchy = pickle.load(open('modelhierarchy.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
pca = pickle.load(open('pca.pkl', 'rb'))


nltk.download('stopwords')
nltk.download('punkt')


ps = PorterStemmer()


def preprocess_text(text):
    text = text.lower()

    text = nltk.word_tokenize(text)

    y = []

    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]

    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]

    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)



class inputModelSingleNewsText(BaseModel):
    inputTextSingle: str


class inputModelMultipleNewsText(BaseModel):
    input_text_one: str
    input_text_two: str
    input_text_three: str 



@app.get('/')
def welcome():
    return {
        'success': True,
        'message': 'server of "fake news clustering is up and running successfully"'
    }


@app.post('/predict-single-news-cluster')
def predict_single_news(inputFromUser: inputModelSingleNewsText):

    userInput = inputFromUser.inputTextSingle

    preproccessed_input = preprocess_text(userInput)

    input_text_vectorized = vectorizer.transform([preproccessed_input]).toarray()

    input_text_vectorized_pca = pca.transform(input_text_vectorized)

    prediction = model_kmeans.predict(input_text_vectorized_pca)

    pred_message = f'The news provided belongs to Cluster Number: {prediction[0]}'

    return {
        'success': True,
        'pred_result': pred_message
    }


@app.post('/predict-multiple-news-cluster')
def predict_multiple_news(inputsFromUser: inputModelMultipleNewsText):

    input_text_input_one = inputsFromUser.input_text_one
    input_text_input_two = inputsFromUser.input_text_two
    input_text_input_three = inputsFromUser.input_text_three 

    input_text_one_preproccessed = preprocess_text(input_text_input_one)
    input_text_two_preproccessed = preprocess_text(input_text_input_two)
    input_text_three_preproccessed = preprocess_text(input_text_input_three)

    input_texts_final = [input_text_one_preproccessed, input_text_two_preproccessed, input_text_three_preproccessed]

    inputs_texts_final_vectorized = vectorizer.transform(input_texts_final).toarray()

    input_texts_final_pca = pca.transform(inputs_texts_final_vectorized)

    prediction = model_hierarchy.fit_predict(input_texts_final_pca)

    pred_result_one = prediction[0]
    pred_result_two = prediction[1]
    pred_result_three = prediction[2]

    pred_message = f'The first news text belongs to Cluster Number: {pred_result_one}. The second news belongs to Cluster Number: {pred_result_two} and the third news belongs to Cluster Number: {pred_result_three}.'

    print(prediction)

    return {
        'success': True,
        'pred_result': pred_message
    }