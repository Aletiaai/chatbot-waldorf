from flask import Flask, request
import vertexai
from vertexai.language_models import ChatModel, TextEmbeddingModel
import ast
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances_argmin as distances_argmin
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from google.cloud import storage
from google.cloud import secretmanager

app = Flask(__name__)
def access_secret_version(project_id, secret_id):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")
# twilio Auth token
project_id = "699244565528"
secret_id = "Twilio_Auth_Token"
auth_token = access_secret_version(project_id, secret_id)
# twilio account SID under the same project ID
secret_id = "Twilio_Account_SID"
account_sid = access_secret_version(project_id, secret_id)
# Initialize Twilio client
client = Client(account_sid, auth_token)
# Initialize Google Cloud Storage client
storage_client = storage.Client()
PROJECT_ID = "chatbot-test-428223"  
LOCATION = "us-central1"  
vertexai.init(project=PROJECT_ID, location=LOCATION)

all_books_db = pd.read_csv('gs://chatbot-waldor/merged_file.csv')

def create_session():
    chat_model = ChatModel.from_pretrained("chat-bison@001")
    chat = chat_model.start_chat()
    return chat

def response(chat, message):
    parameters = {
        "temperature": 0.2,
        "max_output_tokens": 256,
        "top_p": 0.8,
        "top_k": 40
    }
    result = chat.send_message(message, **parameters)
    return result.text

#Take the user query and do the semantic search
def search(user_input_embedding,db):
    all_books = db
    def convert_to_numerical_list(value):
        try:
            numerical_list = ast.literal_eval(value)
            if isinstance(numerical_list, list) and all(isinstance(item, (int, float)) for item in numerical_list):
                return numerical_list
            else:
                return None
        except (SyntaxError, ValueError):
            return None
    # Ensure all_books.Embedding contains numerical lists (optional for clarity)
    if not all(isinstance(x, list) for x in all_books.Embedding):
        all_books['Embedding'] = all_books['Embedding'].apply(convert_to_numerical_list)
    cos_sim_array = cosine_similarity([user_input_embedding], list(all_books.Embedding.values))
    index_doc_cosine = np.argmax(cos_sim_array)
    return index_doc_cosine

#Taking the user query and converting it to embeddings
def input_embeddings(input):
    embedding_model = vertexai.language_models.TextEmbeddingModel.from_pretrained("textembedding-gecko-multilingual@001")
    embedding = embedding_model.get_embeddings([input])[0].values
    return embedding

def complete_input(user_input,book_text, last_fragment_summary):
    #search_result = search(user_input_embedding,db=COMPLETE_DATABASE)
    completemented_input = "Role-playing game:"+\
    "\nYou are Rudolf Steiner, founder of Waldorf education, but you will use the name 'Aletia' instead of Rudolf Steiner."+\
    "\nYou are the mentor of parents who have their children in a Waldorf school. Your goal is to answer all their questions"+\
    "\nrelated to Waldorf education and their children, and help them learn about Waldorf education and how to implement it outside of school with their children."+\
    "\nYou are a polyglot, you are C1 level in English, Spanish and Polish so you answer in the language that the user uses to ask."+\
    "\nIntroduction:"+\
    "\nYou can use this greeting: 'Hello, I am Aletia, and I am here to help you with any questions you have about Waldorf education."+\
    "\nYou can quote literally your work (meaning; Rudolf Steiner's) or those of other relevant authors to support your statements."+\
    "\nYour answers are based on your knowledge (meaning; Rudolf Steiner's) and experience, as well as the information in the database to which you have access."+\
    "\nDatabase:"+\
    "\nInformation Access:'You have access to a vast database of books and articles on Waldorf education, including your own works. This information will be included in every question the user asks'"+\
    "\nTopics: 'You can discuss various topics related to Waldorf education, such as artistic education, the importance of play, child development at different stages of life, the relationship between children and nature, among others.'"+\
    "\nExamples: 'You can provide concrete examples of how Waldorf education is applied in daily life.'"+\
    "\nPersonality:"+\
    "\nKind and patient: 'You are a kind and patient assistant, willing to listen and understand the needs of each parent. Your mission is to explain each question as simply as possible.'"+\
    "\nPassionate: 'You are passionate about education and human development, and you are convinced that Waldorf education can contribute to creating a better world.'"+\
    "\nWise: 'You possess deep wisdom about the human being and their development, which you can share with the user.'"+\
    "\nExamples of questions:"+\
    "\nQuestion: 'What is the importance of play in Waldorf education?'"+\
    "\nAnswer: 'Play is fundamental to a child's development in Waldorf education. It allows the child to explore the world, develop their creativity, imagination, and social skills.'"+\
    "\nQuote: 'Play is the most important work of the child.' - Rudolf Steiner"+\
    "\nQuestion: 'How is reading taught in Waldorf education?'"+\
    "\nAnswer: 'The teaching of reading in Waldorf education is based on a gradual and holistic approach that takes into account the child's development.'"+\
    "\nExample: 'In the early years, the child is introduced to literature through stories, songs, and poems.'"+\
    "\nClosure:"+\
    "\nFarewell: 'It has been a pleasure talking with you. I hope I have helped you understand Waldorf education better.'"+\
    "\nInvitation: 'I invite you to continue learning about Waldorf education and to put it into practice in your daily life.'"+\
    "\nRecommendations:"+\
    "\nUse a warm and kind tone of voice."+\
    "\nBe patient and understanding with the user's questions."+\
    "\nProvide accurate and relevant information."+\
    "\nAlways cite sources of information."+\
    "\nEncourage the user to continue learning about Waldorf education."+\
    "\nTask:"+\
    "\nConsider the following fragment of the database that you have access to, to answer the question: " + book_text + ". " + last_fragment_summary+\
    "\nThe user asked the following question: 'Based on Waldorf education " + user_input + "'"+\
    "\nRemember that your mission is to explain each question to make it cristal clear for the user. Remember also to answer in the language of the following text '" + user_input + "'"
    return completemented_input

def message_classifier_prompt(user_message):
    kind_message = "Background:\n"+\
    "\nYou are an expert classifying messages in three different categories: greeting, question, nothing\n"+\
    "\nyou have a C2 level in English, Spanish and Polish. \n"+\
    "\nThe user will give you a message, it may be something similar to the following examples. I included their classification:\n"+\
    "\nEjemplo 1: 'hola, como estas?', 'Hello how are you?' or 'Cześć, jak się masz?'. classification: 'greeting'"+\
    "\nEjemplo 2: 'hola', 'Hej' or 'hi'. classification: 'greeting'"+\
    "\nEjemplo 3: 'cómo estas?', 'jak się masz?'or 'how is it going?', . classification: 'greeting'"+\
    "\nEjemplo 4: 'como darle continuidad en mi casa a la educación que recibe mi hijo?', 'Jak mogę kontynuować edukację mojego syna w domu?' or 'How can I continue the education my son receives at home?'. classification: 'question'"+\
    "\nEjemplo 5: 'Cuáles son las diferencias entre las pedagogías Waldorf y la Montessori?', 'Jakie są różnice między pedagogiką Waldorf i Montessori?' or 'What are the differences between Waldorf and Montessori pedagogies?'. classification: 'question'"+\
    "\nEjemplo 6: 'que actividades puedo hacer en casa para seguir con la pedagogía Waldorf', 'What activities can I do at home to continue with Waldorf pedagogy?' or 'Jakie działania mogę podjąć w domu, aby kontynuować pedagogikę Waldorf?'. classification: 'question'"+\
    "\nEjemplo 7: 'qué tipo de lecturas ayudan a mi hijo a dormir', 'What type of readings help my child to sleep?' or 'Jakie rodzaje czytań pomagają mojemu dziecku zasnąć?'. classification: 'question'"+\
    "\nEjemplo 11: '?'. classification: 'nothing'"+\
    "\nEjemplo 12: '. classification: 'nothing'"+\
    "\nTask:\n"+\
    "\Classify the following user input in the following options: 'question', 'nothing' o 'greeting'. Texto: '" + user_message + "'. Only answer with the classification of the message."
    return kind_message

def existing_user_response(chat,response_classification,user_question):
    if response_classification == "question":
        user_embedding = input_embeddings(user_question)
        search_result = search(user_embedding,all_books_db)
        book_fragment = all_books_db.Content[search_result]
        last_summary = str(all_books_db.Summary[search_result])
        complemented_question = complete_input(user_question, book_fragment, last_summary)
        response_generated = response(chat, complemented_question)
        response_generated = response_generated + "\n\nReferencia: "+ all_books_db.Title[search_result]+ " por " + str(all_books_db.Author[search_result])
        return response_generated
    elif response_classification == "greeting":
        how_i_can_help_message = "Hello, How can I help you today? you can ask in Polish, Spanish or English 🤓"
        return how_i_can_help_message
    elif response_classification == "nothing":
        not_sure_message = "I am sorry, I didn't get your last message 🫣"
        return not_sure_message
@app.route('/base', methods=['GET', 'POST'])
def base():
        user_input = ""
        if request.method == 'GET':
            user_input = request.args.get('Body', '')
            phone_number = request.args.get('From', '') #Including the prefix "whatsapp:""
        else:
            user_input = request.form['Body']
            phone_number = request.form['From'] #Including the prefix "whatsapp:""
        
        chat_model = create_session()
         #Identifying what kind of message did the user sent: question, nothing or greating.
        kindof_message_prompt = message_classifier_prompt(user_input)
        message_classfication = response(chat_model, kindof_message_prompt)
        response_to_user = existing_user_response(chat_model,message_classfication,user_input)
        twiml_response = MessagingResponse()
        twiml_response.message(response_to_user)
        return str(twiml_response)    

if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')