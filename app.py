import builtins
import io
from flask import Flask, request
import vertexai
from vertexai.language_models import ChatModel, TextEmbeddingModel
import os
import ast
import numpy as np
import pandas as pd
import firebase_admin
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances_argmin as distances_argmin
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from firebase_admin import firestore, credentials
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
# Access the bucket
bucket_name = 'chatbot-waldor'
database_file = 'users_database.csv'
blob = storage_client.bucket(bucket_name).blob(database_file)
# Load the DataFrame from the CSV content in Google Cloud Storage
database_df = pd.read_csv(blob.open())
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
def get_name1(complete_user_message, chat):
    prompt = ""
    prompt = "Background:\n"+\
    "\nEl usuario ingresará un texto, los siguientes son ejemplos de texto ingresados por el usuario:\n"+\
    "\nEjemplo 1: Marco Polo García Martínez, hsnetl@gmail.com\n"+\
    "\nEl Ejemplo 1 es correcto ya que el primer nombre: Marco, Segundo nombre: Polo, apellido paterno: García, apellido materno: Martínez, correo electrónico: hsnetl@gmail.com\n"+\
    "\nEjemplo 2: Marco García Martínez, hsnetl38829@outlook.com\n"+\
    "\nEl Ejemplo 2 es correcto ya que el primer nombre: Marco, Segundo nombre: no lo ingresó pero es posible que no tenga por lo tanto solo rellena este espacio con 'na', apellido paterno: García, apellido materno: Martínez, correo electrónico: hsnetl38829@outlook.com\n"+\
    "\nEjemplo 3: Carmen Martínez Rodríguez, hsdm5439_0l@gmail.com\n"+\
    "\nEl Ejemplo 3 es correcto ya que el primer nombre: Carmen, Segundo nombre: no lo ingresó pero es posible que no tenga por lo tanto solo rellena este espacio con 'na', apellido paterno: Martínez, apellido materno: Rodríguez, correo electrónico: hsdm5439_0l@gmail.com\n"+\
    "\nEjemplo 4: Martínez Rodríguez, hsdm5439_0l@gmail.com\n"+\
    "\nEl Ejemplo 4 es incorrecto ya que el primer nombre: no lo ingreso el usuario, Segundo nombre: no lo ingresó pero es posible que no tenga por lo tanto solo rellena este espacio con 'na', apellido paterno: Martínez, apellido materno: Rodríguez, correo electrónico: hsdm5439_0l@gmail.com.\n"+\
    "\nEjemplo 5: hsdm5439_0l@gmail.com\n"+\
    "\nEl Ejemplo 5 es incorrecto ya que el 'primer nombre': no lo ingresó el usuario, 'Segundo nombre': no lo ingresó el usuario, 'apellido paterno': no lo ingresó el usuario, 'apellido materno': no lo ingresó el usuario, correo electrónico: hsdm5439_0l@gmail.com.\n"+\
    "\nEjemplo 6: Marco Polo García Martínez\n"+\
    "\nEl Ejemplo 6 es incorrecto ya que el 'primer nombre': Marco, 'Segundo nombre': Polo, 'apellido paterno': García, 'apellido materno': Martínez, correo electrónico: no lo ingresó el usuario.\n"+\
    "\nTask:\n"+\
    "\nExtrae el primer nombre del usuario del siguiente texto: "+ complete_user_message + ". Si no encuentras el primer nombre solo respondeme con 'na'"
    name1_data = response(chat, prompt)
    return name1_data
def get_name2(complete_user_message, chat):
    prompt = ""
    prompt = "Background:\n"+\
    "\nEl usuario ingresará un texto, los siguientes son ejemplos de texto ingresados por el usuario:\n"+\
    "\nEjemplo 1: Marco Polo García Martínez, hsnetl@gmail.com\n"+\
    "\nEl Ejemplo 1 es correcto ya que el primer nombre: Marco, Segundo nombre: Polo, apellido paterno: García, apellido materno: Martínez, correo electrónico: hsnetl@gmail.com\n"+\
    "\nEjemplo 2: Marco García Martínez, hsnetl38829@outlook.com\n"+\
    "\nEl Ejemplo 2 es correcto ya que el primer nombre: Marco, Segundo nombre: no lo ingresó pero es posible que no tenga por lo tanto solo rellena este espacio con 'na', apellido paterno: García, apellido materno: Martínez, correo electrónico: hsnetl38829@outlook.com\n"+\
    "\nEjemplo 3: Carmen Martínez Rodríguez, hsdm5439_0l@gmail.com\n"+\
    "\nEl Ejemplo 3 es correcto ya que el primer nombre: Carmen, Segundo nombre: no lo ingresó pero es posible que no tenga por lo tanto solo rellena este espacio con 'na', apellido paterno: Martínez, apellido materno: Rodríguez, correo electrónico: hsdm5439_0l@gmail.com\n"+\
    "\nEjemplo 4: Martínez Rodríguez, hsdm5439_0l@gmail.com\n"+\
    "\nEl Ejemplo 4 es incorrecto ya que el primer nombre: no lo ingreso el usuario, Segundo nombre: no lo ingresó pero es posible que no tenga por lo tanto solo rellena este espacio con 'na', apellido paterno: Martínez, apellido materno: Rodríguez, correo electrónico: hsdm5439_0l@gmail.com.\n"+\
    "\nEjemplo 5: hsdm5439_0l@gmail.com\n"+\
    "\nEl Ejemplo 5 es incorrecto ya que el 'primer nombre': no lo ingresó el usuario, 'Segundo nombre': no lo ingresó el usuario, 'apellido paterno': no lo ingresó el usuario, 'apellido materno': no lo ingresó el usuario, correo electrónico: hsdm5439_0l@gmail.com.\n"+\
    "\nEjemplo 6: Marco Polo García Martínez\n"+\
    "\nEl Ejemplo 6 es incorrecto ya que el 'primer nombre': Marco, 'Segundo nombre': Polo, 'apellido paterno': García, 'apellido materno': Martínez, correo electrónico: no lo ingresó el usuario.\n"+\
    "\nTask:\n"+\
    "\nExtrae el segundo nombre del usuario del siguiente texto: "+ complete_user_message + ". Si no encuentras el segundo nombre solo respondeme con 'na'"
    name2_data = response(chat, prompt)
    return name2_data
def get_lastname1(complete_user_message, chat):
    prompt = ""
    prompt = "Background:\n"+\
    "\nEl usuario ingresará un texto, los siguientes son ejemplos de texto ingresados por el usuario:\n"+\
    "\nEjemplo 1: Marco Polo García Martínez, hsnetl@gmail.com\n"+\
    "\nEl Ejemplo 1 es correcto ya que el primer nombre: Marco, Segundo nombre: Polo, apellido paterno: García, apellido materno: Martínez, correo electrónico: hsnetl@gmail.com\n"+\
    "\nEjemplo 2: Marco García Martínez, hsnetl38829@outlook.com\n"+\
    "\nEl Ejemplo 2 es correcto ya que el primer nombre: Marco, Segundo nombre: no lo ingresó pero es posible que no tenga por lo tanto solo rellena este espacio con 'na', apellido paterno: García, apellido materno: Martínez, correo electrónico: hsnetl38829@outlook.com\n"+\
    "\nEjemplo 3: Carmen Martínez Rodríguez, hsdm5439_0l@gmail.com\n"+\
    "\nEl Ejemplo 3 es correcto ya que el primer nombre: Carmen, Segundo nombre: no lo ingresó pero es posible que no tenga por lo tanto solo rellena este espacio con 'na', apellido paterno: Martínez, apellido materno: Rodríguez, correo electrónico: hsdm5439_0l@gmail.com\n"+\
    "\nEjemplo 4: Martínez Rodríguez, hsdm5439_0l@gmail.com\n"+\
    "\nEl Ejemplo 4 es incorrecto ya que el primer nombre: no lo ingreso el usuario, Segundo nombre: no lo ingresó pero es posible que no tenga por lo tanto solo rellena este espacio con 'na', apellido paterno: Martínez, apellido materno: Rodríguez, correo electrónico: hsdm5439_0l@gmail.com.\n"+\
    "\nEjemplo 5: hsdm5439_0l@gmail.com\n"+\
    "\nEl Ejemplo 5 es incorrecto ya que el 'primer nombre': no lo ingresó el usuario, 'Segundo nombre': no lo ingresó el usuario, 'apellido paterno': no lo ingresó el usuario, 'apellido materno': no lo ingresó el usuario, correo electrónico: hsdm5439_0l@gmail.com.\n"+\
    "\nEjemplo 6: Marco Polo García Martínez\n"+\
    "\nEl Ejemplo 6 es incorrecto ya que el 'primer nombre': Marco, 'Segundo nombre': Polo, 'apellido paterno': García, 'apellido materno': Martínez, correo electrónico: no lo ingresó el usuario.\n"+\
    "\nTask:\n"+\
    "\nExtrae el apellido paterno del usuario del siguiente texto: "+ complete_user_message + ". Si no encuentras el apellido paterno solo respondeme con 'na'"
    lastname1_data = response(chat, prompt)
    return lastname1_data
def get_lastname2(complete_user_message, chat):
    prompt = ""
    prompt = "Background:\n"+\
    "\nEl usuario ingresará un texto, los siguientes son ejemplos de texto ingresados por el usuario:\n"+\
    "\nEjemplo 1: Marco Polo García Martínez, hsnetl@gmail.com\n"+\
    "\nEl Ejemplo 1 es correcto ya que el primer nombre: Marco, Segundo nombre: Polo, apellido paterno: García, apellido materno: Martínez, correo electrónico: hsnetl@gmail.com\n"+\
    "\nEjemplo 2: Marco García Martínez, hsnetl38829@outlook.com\n"+\
    "\nEl Ejemplo 2 es correcto ya que el primer nombre: Marco, Segundo nombre: no lo ingresó pero es posible que no tenga por lo tanto solo rellena este espacio con 'na', apellido paterno: García, apellido materno: Martínez, correo electrónico: hsnetl38829@outlook.com\n"+\
    "\nEjemplo 3: Carmen Martínez Rodríguez, hsdm5439_0l@gmail.com\n"+\
    "\nEl Ejemplo 3 es correcto ya que el primer nombre: Carmen, Segundo nombre: no lo ingresó pero es posible que no tenga por lo tanto solo rellena este espacio con 'na', apellido paterno: Martínez, apellido materno: Rodríguez, correo electrónico: hsdm5439_0l@gmail.com\n"+\
    "\nEjemplo 4: Martínez Rodríguez, hsdm5439_0l@gmail.com\n"+\
    "\nEl Ejemplo 4 es incorrecto ya que el primer nombre: no lo ingreso el usuario, Segundo nombre: no lo ingresó pero es posible que no tenga por lo tanto solo rellena este espacio con 'na', apellido paterno: Martínez, apellido materno: Rodríguez, correo electrónico: hsdm5439_0l@gmail.com.\n"+\
    "\nEjemplo 5: hsdm5439_0l@gmail.com\n"+\
    "\nEl Ejemplo 5 es incorrecto ya que el 'primer nombre': no lo ingresó el usuario, 'Segundo nombre': no lo ingresó el usuario, 'apellido paterno': no lo ingresó el usuario, 'apellido materno': no lo ingresó el usuario, correo electrónico: hsdm5439_0l@gmail.com.\n"+\
    "\nEjemplo 6: Marco Polo García Martínez\n"+\
    "\nEl Ejemplo 6 es incorrecto ya que el 'primer nombre': Marco, 'Segundo nombre': Polo, 'apellido paterno': García, 'apellido materno': Martínez, correo electrónico: no lo ingresó el usuario.\n"+\
    "\nTask:\n"+\
    "\nExtrae el apellido materno del usuario del siguiente texto: "+ complete_user_message + ". Si no encuentras el apellido materno solo respondeme con 'na'"
    lastname2_data = response(chat, prompt)
    return lastname2_data
def get_mail(complete_user_message, chat):
    prompt = ""
    prompt = "Background:\n"+\
    "\nEl usuario ingresará un texto, los siguientes son ejemplos de texto ingresados por el usuario:\n"+\
    "\nEjemplo 1: Marco Polo García Martínez, hsnetl@gmail.com\n"+\
    "\nEl Ejemplo 1 es correcto ya que el primer nombre: Marco, Segundo nombre: Polo, apellido paterno: García, apellido materno: Martínez, correo electrónico: hsnetl@gmail.com\n"+\
    "\nEjemplo 2: Marco García Martínez, hsnetl38829@outlook.com\n"+\
    "\nEl Ejemplo 2 es correcto ya que el primer nombre: Marco, Segundo nombre: no lo ingresó pero es posible que no tenga por lo tanto solo rellena este espacio con 'na', apellido paterno: García, apellido materno: Martínez, correo electrónico: hsnetl38829@outlook.com\n"+\
    "\nEjemplo 3: Carmen Martínez Rodríguez, hsdm5439_0l@gmail.com\n"+\
    "\nEl Ejemplo 3 es correcto ya que el primer nombre: Carmen, Segundo nombre: no lo ingresó pero es posible que no tenga por lo tanto solo rellena este espacio con 'na', apellido paterno: Martínez, apellido materno: Rodríguez, correo electrónico: hsdm5439_0l@gmail.com\n"+\
    "\nEjemplo 4: Martínez Rodríguez, hsdm5439_0l@gmail.com\n"+\
    "\nEl Ejemplo 4 es incorrecto ya que el primer nombre: no lo ingreso el usuario, Segundo nombre: no lo ingresó pero es posible que no tenga por lo tanto solo rellena este espacio con 'na', apellido paterno: Martínez, apellido materno: Rodríguez, correo electrónico: hsdm5439_0l@gmail.com.\n"+\
    "\nEjemplo 5: hsdm5439_0l@gmail.com\n"+\
    "\nEl Ejemplo 5 es incorrecto ya que el 'primer nombre': no lo ingresó el usuario, 'Segundo nombre': no lo ingresó el usuario, 'apellido paterno': no lo ingresó el usuario, 'apellido materno': no lo ingresó el usuario, correo electrónico: hsdm5439_0l@gmail.com.\n"+\
    "\nEjemplo 6: Marco Polo García Martínez\n"+\
    "\nEl Ejemplo 6 es incorrecto ya que el 'primer nombre': Marco, 'Segundo nombre': Polo, 'apellido paterno': García, 'apellido materno': Martínez, correo electrónico: no lo ingresó el usuario.\n"+\
    "\nTask:\n"+\
    "\nExtrae el correo electrónico del usuario del siguiente texto: "+ complete_user_message + ". Si no encuentras el correo electrónico solo respondeme con 'na'"
    email_data = response(chat, prompt)
    return email_data
def user_exists(phone_number, df):
    # Check if the phone number exists in the DataFrame
    return builtins.any(df['phone'] == phone_number)
def update_answers_sent(phone_number,db):    
    # Get the index of the user in the DataFrame
    user_index = db.index[db['phone'] == phone_number][0]
        
    # Increment the value of answers_sent for the user
    db.at[user_index, 'answers_sent'] += 1
    answers = db.at[user_index, 'answers_sent']
        
    # Write the updated DataFrame back to the CSV file in the bucket
    with io.StringIO() as csv_buffer:
        db.to_csv(csv_buffer, index=False)
        blob.upload_from_string(csv_buffer.getvalue())
        return answers
def updated_answers_sent(phone_number,db):    
    # Get the index of the user in the DataFrame
    user_index = db.index[db['phone'] == phone_number][0]
    # Get the updated value from "answers_sent" column
    answers_updated = db.at[user_index, 'answers_sent']
    return answers_updated
def update_current_question(phone_number,db):    
    # Get the index of the user in the DataFrame
    user_index = db.index[db['phone'] == phone_number][0]
    # Increment the value of answers_sent for the user
    db.at[user_index, 'current_question'] += 1
    question_asked = db.at[user_index, 'current_question']

    # Write the updated DataFrame back to the CSV file in the bucket
    with io.StringIO() as csv_buffer:
        db.to_csv(csv_buffer, index=False)
        blob.upload_from_string(csv_buffer.getvalue())
    return question_asked
def updated_current_question(phone_number,db):    
    # Get the index of the user in the DataFrame
    user_index = db.index[db['phone'] == phone_number][0]
    # Get the updated value of "current_question" column
    question_asked_updated = db.at[user_index, 'current_question']
    return question_asked_updated
def update_total_questions(phone_number,db):    
    # Get the index of the user in the DataFrame
    user_index = db.index[db['phone'] == phone_number][0]
        
    # Increment the value of answers_sent for the user
    db.at[user_index, 'total_questions'] += 1
    total_questions = db.at[user_index, 'total_questions']
        
    # Write the updated DataFrame back to the CSV file in the bucket
    with io.StringIO() as csv_buffer:
        db.to_csv(csv_buffer, index=False)
        blob.upload_from_string(csv_buffer.getvalue())
    return total_questions
def updated_total_questions(phone_number,db):    
    # Get the index of the user in the DataFrame
    user_index = db.index[db['phone'] == phone_number][0]
    # Get the updated value of "total_questions" column
    total_questions_updated = db.at[user_index, 'total_questions']
    return total_questions_updated
def select_question(current_question, chat):
    prompt = "La siguiente es una lista de preguntas:\n"+\
    "\ntest_1_quest_1 = '¿Qué es la pedagogía Waldorf?'\n"+\
    "\ntest_1_quest_2 = '¿Cuáles son los principios fundamentales de la pedagogía Waldorf?'\n"+\
    "\ntest_1_quest_3 = '¿En qué se diferencia la pedagogía Waldorf de la educación tradicional?'\n"+\
    "\ntest_1_quest_4 = 'Tu hijo de 2 años tiene una rabieta en el supermercado. ¿Cómo aplicarías la pedagogía Waldorf para manejar esta situación?'\n"+\
    "\ntest_1_quest_5 = 'Tu hija de 4 años te pregunta de dónde vienen los bebés. ¿Cómo le explicarías este tema de acuerdo a la pedagogía Waldorf?'\n"+\
    "\ntest_1_quest_6 = '¿Qué tipo de actividades puedes realizar en casa con tus hijos para fomentar su desarrollo creativo e imaginativo, siguiendo los principios de la pedagogía Waldorf?'\n"+\
    "\ntest_1_quest_7 = '¿Cuáles son las principales ventajas de la pedagogía Waldorf?'\n"+\
    "\ntest_1_quest_8 = '¿Cuáles son las principales desventajas de la pedagogía Waldorf?'\n"+\
    "\ntest_1_quest_9 = '¿Consideras que la pedagogía Waldorf es adecuada para tus hijos? ¿Por qué?'\n"+\
    "\nExtrae solo la pregunta correspondiente a '" + current_question + "' y envíamela"
    next_question = response(chat, prompt)
    return next_question
def message_classifier_prompt(user_message):
    kind_message = "Background:\n"+\
    "\nEl usuario ingresará un texto, los siguientes son ejemplos de texto ingresados por el usuario y su clasificación:\n"+\
    "\nEjemplo 1: 'hola, como estas?'. Clasificación: 'saludo'"+\
    "\nEjemplo 2: 'hola'. Clasificación: 'saludo'"+\
    "\nEjemplo 3: 'cómo estas?'. Clasificación: 'saludo'"+\
    "\nEjemplo 4: 'como darle continuidad en mi casa a la educación que recibe mi hijo?'. Clasificación: 'pregunta'"+\
    "\nEjemplo 5: 'Cuáles son las diferencias entre las pedagogías más famosas?'. Clasificación: 'pregunta'"+\
    "\nEjemplo 6: 'que actividades puedo hacer en casa para seguir con la pedagogía'"+\
    "\nEjemplo 7: 'qué tipo de lecturas yudan a mi hijo a dormir'. Clasificación: 'pregunta'"+\
    "\nEjemplo 8: 'la pedagogía Waldorf es una pedagogía alternativa y desde mi punto de vista tiene tiene algunas ventajas bastante importantes si la comparamos con la pedagogía tradicional'. Clasificación: 'respuesta'"+\
    "\nEjemplo 9: 'Si mi hija esta haciendo un berrince en el supermercado entonces me hinco para estar a la misma altura que ella y trato de validar sus emmociones e intento abrazarla.'. Clasificación: 'respuesta'"+\
    "\nEjemplo 10: 'Es una pedagogía diferente a la tradicional porque motiva a los niños a ser independientes y ser conscientes de sus emociones'. Clasificación: 'respuesta'"+\
    "\nEjemplo 11: '?'. Clasificación: 'nada'"+\
    "\nEjemplo 12: '. Clasificación: 'nada'"+\
    "\nTask:\n"+\
    "\nClasifica el siguiente texto en 'pregunta', 'respuesta', 'nada' o 'saludo'. Texto: '" + user_message + "'. Solo responde con la clasificación."
    return kind_message
def prompt_suggested_questions (chat,tel_number,user_answer):
    test_questions_asked = int(updated_current_question(tel_number, database_df))-1
    current_question = "test_1_quest_"+ str(test_questions_asked)
    text_current_question = select_question(current_question, chat)
    prompt = "Juego de roles:"+\
    "\nEres Rudolf Steiner, fundador de la pedagogía Waldorf pero vas a usar el nombre de 'Aletia' en lugar del de Rudolf Steiner.\n"+\
    "\nEres el mentor de padres y madres que tienen a sus hijos o hijas en una escuela Waldorf. Tu objetivo es ayudar a estos padres y madres a aprender sobre la pedagogía Waldorf "+\
    "\ny como implementarla fuera de la escuela en sus hijos.\n"+\
    "\nTarea: Analiza la respuesta del usuario y sugierele 3 preguntas que puede hacerte a ti, Rudolf Steiner, para que cuando le des la respuesta, el usuario aprender más de los temas involucrados en la pregunta que le hiciste.\n"+\
    "\nRecuerda, la pregunta que le hiciste fue:"+ text_current_question + ".\n"+\
    "\ny la respuesta del usuario fue :"+ user_answer + "."+\
    "\nEjemplos de preguntas son:.\n"+\
    "\n1. ¿Cuáles son los pilares de la pedagogía Wzldorf?\n"+\
    "\n2. ¿Cómo logra la pedagogía Waldorf que los niños sean independientes?\n"+\
    "\n3. ¿Puedes explicarme a que significa que la pedagogía Wldorf sea 'libre'?\n"+\
    "\n4. ¿Qué tipo de juegos e utilizan en la pedagogía Waldorf para foprtalezar la seguridad en los niños?\n"+\
    "\nContesta solo con las 3 preguntas sugeridas"
    return prompt
def existing_user_response(chat,response_classification,test_questions_asked,answers_given,user_question,tel_number):
    test_length = 4
    if response_classification == "pregunta":
        if test_questions_asked == answers_given:
            if answers_given >= test_length:
                total_questions = update_total_questions(tel_number, database_df)
                user_embedding = input_embeddings(user_question)
                search_result = search(user_embedding,all_books_db)
                book_fragment = all_books_db.Content[search_result]
                last_summary = str(all_books_db.Summary[search_result])
                complemented_question = complete_input(user_question, book_fragment, last_summary)
                response_generated = response(chat, complemented_question)
                response_generated = response_generated + "\n\nReferencia: "+ all_books_db.Title[search_result]+ " por " + str(all_books_db.Author[search_result])
                return response_generated
            else:
                test_questions_asked = update_current_question(tel_number, database_df)
                total_questions = update_total_questions(tel_number, database_df)
                current_question = "test_1_quest_"+ str(test_questions_asked)
                next_question = select_question(current_question, chat)
                user_embedding = input_embeddings(user_question)
                search_result = search(user_embedding,all_books_db)
                book_fragment = all_books_db.Content[search_result]
                last_summary = str(all_books_db.Summary[search_result])
                complemented_question = complete_input(user_question, book_fragment, last_summary)
                response_generated = response(chat, complemented_question)
                response_generated = response_generated + "\n\nReferencia: "+ all_books_db.Title[search_result]+ " por " + str(all_books_db.Author[search_result]) + "\n\nLa siguiente pregunta que quiero hacerte es:\n" + str(test_questions_asked) + ". " + next_question
                return response_generated
        else:
            quest_dif_toans_message = "Sé que tienes preguntas y para eso estoy aquí 😎, pero necesito que contestes la pregunta anterior, por favor 🫣."
            return quest_dif_toans_message   
    elif response_classification == "saludo":
        if test_questions_asked == answers_given:
            how_i_can_help_message = "Hola, ¿con qué pregunta puedo ayudarte? 🤓"
            return how_i_can_help_message
        elif test_questions_asked > answers_given:
            answer_last_question_message = "Hola, es importante que contentes la pregunta anterior... por favor 😕"        
            return answer_last_question_message
    elif response_classification == "respuesta":
        answers_given = update_answers_sent(tel_number, database_df)
        prompt_to_get_questions = prompt_suggested_questions (chat, tel_number,user_question)
        three_suggested_questions = response(chat, prompt_to_get_questions)
        sugestion = "Ya tengo tu respuesta 🥳, gracias!.\n"+\
        "\nAhora es tu turno, puedes hacerme cualquier pregunta sobre la pedagogía Waldorf 🤓.\n"+\
        "\nAlgunas sugerencias de acuerdo a tu respuesta anterior son:\n"+\
        "\n\n" + three_suggested_questions
        return sugestion
    elif response_classification == "nada":
        not_sure_message = "Lo siento, no entendí tu último mensaje 🫣"
        return not_sure_message
@app.route('/wasupp', methods=['GET', 'POST'])
def wasupp():
        chat_model = create_session()
        global database_df  # Declare database_df as a global variable
        test_1_quest_1 = "¿Qué es la pedagogía Waldorf?"
        user_input = ""
        name1 = "na"
        name2 = "na"
        lastname1 = "na"
        lastname2 = "na"
        personal_email = "na"
        if request.method == 'GET':
            user_input = request.args.get('Body', '')
            phone_number = request.args.get('From', '') #Including the prefix "whatsapp:""
        else:
            user_input = request.form['Body']
            phone_number = request.form['From'] #Including the prefix "whatsapp:""
        if not phone_number:
            return "Error: No phone number provided", 400
        number = phone_number.replace("whatsapp:+", "") # Remove "whatsapp:" prefix from phone number
        name1 = get_name1(user_input,chat_model)
        name2 = get_name2(user_input,chat_model)
        lastname1 = get_lastname1(user_input,chat_model)
        lastname2 = get_lastname2(user_input,chat_model)
        personal_email = get_mail(user_input,chat_model)
        if not user_exists(number, database_df):
            # New user
            if name1 == "na" or lastname1 == "na" or lastname2 == "na" or personal_email == "na":
                ask_message = "Hola, me da gusto verte por aquí, soy Aletia y para responder a tus preguntas de la pedagogía Waldorf necesito crear tu cuenta.\n\nIngresa tu nombre completo y tu correo electrónico, por favor.\n\nEjemplo: Juan Carlos Pérez Nájera, jcarlos56@hotmail.com\n\nSi falta algun dato volveras a recibir este mensaje."
                database_df['phone'] = database_df['phone'].astype(str)
                twiml_response = MessagingResponse()
                twiml_response.message(ask_message)
                return str(twiml_response)
            else:
                # Create a dictionary with new user data
                new_user_data = {'phone': number, 'name_1': name1, 'name_2': name2, 'last_name_1': lastname1, 'last_name_2': lastname2, 'email': personal_email,'current_question': 0, 'total_questions': 0, 'answers_sent': 0,}
                # Convert the dictionary into a DataFrame
                new_user_df = pd.DataFrame([new_user_data])
                # Concatenate the new user DataFrame with the existing database DataFrame
                database_df = pd.concat([database_df, new_user_df], ignore_index=True)
                # Save the updated DataFrame back to the CSV file in the bucket
                with io.StringIO() as csv_buffer:
                    database_df.to_csv(csv_buffer, index=False)
                    csv_content = csv_buffer.getvalue().encode('utf-8')
                    blob.upload_from_string(csv_content, content_type='text/csv')
                welcome_message = "envía el siguiente mensaje y personalizalo de acuerdo al género de "+ name1 + "\n"+\
                "\n'🌟 ¡Bienvenido "+ name1 +", soy Aletia, tu guía para la educación Waldorf! 🌟\n"+\
                "\nEstoy encantada de tenerte aquí en este viaje de descubrimiento y apoyo para la educación de tu hijo/a. 😊\n"+\
                "\nPara adaptar mis respuestas a tus necesidades, me encantaría hacerte 4 preguntas, la primer pregunta es:\n\n"+\
                "\n1. "+ test_1_quest_1 +"\n\n"+\
                "\nTu respuesta me ayudará a personalizar tu experiencia. 🌿✨'"
                question_asked = update_current_question(number, database_df) #increasing the variable question_asked since I sent the 1st question already
                answers =  updated_answers_sent(number,database_df)# Updating values for answers variable
                question_asked = updated_current_question(number,database_df) # Updating values for question_asked variable 
                total_questions = updated_total_questions(number,database_df) # Updating values for total_question variable 
                account_created = response(chat_model, welcome_message)
                twiml_response = MessagingResponse()
                twiml_response.message(account_created)
                return str(twiml_response)
        else:
            answers =  updated_answers_sent(number,database_df)# Updating values for answers variable
            question_asked = updated_current_question(number,database_df) # Updating values for question_asked variable 
            total_questions = updated_total_questions(number,database_df) # Updating values for total_question variable 
            # Existing user, identify what kind of message did the user sent: question, answer, nothing or greating.
            kindof_message_prompt = message_classifier_prompt(user_input)
            message_classfication = response(chat_model, kindof_message_prompt)
            response_to_user = existing_user_response(chat_model,message_classfication,question_asked,answers,user_input,number)
            twiml_response = MessagingResponse()
            twiml_response.message(response_to_user)
            return str(twiml_response)        
if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')