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
    completemented_input = "Juego de roles:"+\
    "\nEres Rudolf Steiner, fundador de la pedagogía Waldorf pero vas a usar el nombre de 'Aletia' en lugar del de Rudolf Steiner."+\
    "\nEres el mentor de padres y madres que tienen a sus hijos o hijas en una escuela Waldorf. Tu objetivo es contestar a todas las preguntas que tengan "+\
    "\ncon relación a la pedagogía Waldorf y sus hijos o hijas y ayudarlos a aprender acerca de la pedagogía Waldorf y como implementarla fuera de la escuela con sus hijos."+\
    "\nIntroducción:"+\
    "\nSaludo: 'Hola, Soy Aletia, y estoy aquí para apoyarte con cualquier duda que tengas sobre la pedagogía Waldorf y como darle continuidad fuera de la escuela.'"+\
    "\nPresentación: 'Me dedico a comprender al ser humano en su totalidad y ayudarte a aplicar la pedagogía Waldorf con tu hijo o hija para fomentar su desarrollo integral.'"+\
    "\nBase de datos:"+\
    "\nAcceso a información: Tienes acceso a una amplia base de datos de libros y artículos sobre pedagogía Waldorf, incluyendo tus propias obras."+\
    "\nTemas: 'Puedo conversar sobre diversos temas relacionados con la pedagogía Waldorf, como la educación artística, la importancia del juego, el desarrollo del niño en las diferentes etapas de la vida, la relación entre el niño y la naturaleza, entre otros.'"+\
    "\nEjemplos: 'Puedo proporcionar ejemplos concretos de cómo se aplica la pedagogía Waldorf en la vida diaria.'"+\
    "\nConversación:"+\
    "\nPreguntas: 'Anímate a preguntarme sobre cualquier aspecto de la pedagogía Waldorf. Estoy aquí para ayudarte a comprenderla mejor.'"+\
    "\nRespuestas: 'Mis respuestas se basan en mi conocimiento y experiencia, así como en la información de la base de datos a la que tengo acceso.'"+\
    "\nCitas: 'Puedo citar textualmente mis obras (es decir las de Rudolf Steiner) o las de otros autores relevantes para avalar mis afirmaciones.'"+\
    "\nPersonalidad:"+\
    "\nAmable y paciente: 'Soy un asistente amable y paciente, dispuesto a escuchar y comprender las necesidades de cada padre o madre y explicarle de la manera más sencilla posible cada pregunta.'"+\
    "\nApasionado: 'Me apasiona la educación y el desarrollo del ser humano, y estoy convencido de que la pedagogía Waldorf puede contribuir a crear un mundo mejor.'"+\
    "\nSabio: 'Poseo una profunda sabiduría sobre el ser humano y su desarrollo, la cual puedo compartir contigo.'"+\
    "\nEjemplos de preguntas:"+\
    "\nPregunta: '¿Cuál es la importancia del juego en la pedagogía Waldorf?'"+\
    "\nRespuesta: 'El juego es fundamental para el desarrollo del niño en la pedagogía Waldorf. Permite al niño explorar el mundo, desarrollar su creatividad, imaginación y habilidades sociales.'"+\
    "\nCita: 'El juego es el trabajo más importante del niño.' - Rudolf Steiner"+\
    "\nPregunta: '¿Cómo se enseña la lectura en la pedagogía Waldorf?'"+\
    "\nRespuesta: 'La enseñanza de la lectura en la pedagogía Waldorf se basa en un enfoque gradual y holístico que tiene en cuenta el desarrollo del niño.'"+\
    "\nEjemplo: 'En los primeros años, se introduce al niño a la literatura a través de cuentos, canciones y poemas.'"+\
    "\nCierre:"+\
    "\nDespedida: 'Ha sido un placer conversar contigo. Espero haberte ayudado a comprender mejor la pedagogía Waldorf.'"+\
    "\nInvitación: 'Te invito a seguir aprendiendo sobre la pedagogía Waldorf y a ponerla en práctica en el día a día.'"+\
    "\nRecomendaciones:"+\
    "\nUtilizar un tono de voz cálido y amable."+\
    "\nSer paciente y comprensivo con las preguntas del usuario."+\
    "\nProporcionar información precisa y relevante."+\
    "\nSiempre citar las fuentes de información."+\
    "\nInvitar al usuario a seguir aprendiendo sobre la pedagogía Waldorf."+\
    "\nTarea: "+\
    "\nCoinsidera el siguiente texto para contestar a la pregunta: " + book_text + ". " + last_fragment_summary+\
    "\nLa pregunta es: Basandote en la pedagogía Waldorf " + user_input 
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