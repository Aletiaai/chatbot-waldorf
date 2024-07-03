import os
import vertexai
import pandas as pd
import numpy as np
import ast
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances_argmin as distances_argmin

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

cred = credentials.Certificate("gs://waldorf-db/tough-volt-375400-d8b9ebef5d01.json")
firebase_admin.initialize_app(cred)

# Create a Firestore client
db = firestore.client()

PROJECT_ID = "tough-volt-375400"  
LOCATION = "us-central1" 
COMPLETE_DATABASE = pd.read_csv('/Users/marco/db/merged_file.csv') #gs://waldorf-db/merged_file.csv

vertexai.init(project=PROJECT_ID, location=LOCATION)

def create_session():
    chat_model = vertexai.language_models.ChatModel.from_pretrained("chat-bison@001")
    chat = chat_model.start_chat()
    return chat

#Taking the user query and convertin it to e¡mbeddings
def input_embeddings(input):
    embedding_model = vertexai.language_models.TextEmbeddingModel.from_pretrained("textembedding-gecko-multilingual@001")
    input_embedding = embedding_model.get_embeddings([input])[0].values
    return input_embedding

#Take the user query and do the semantic search
def search(user_input_embedding,db=COMPLETE_DATABASE):
    complete_database = db
    def convert_to_numerical_list(value):
        try:
            numerical_list = ast.literal_eval(value)
            if isinstance(numerical_list, list) and all(isinstance(item, (int, float)) for item in numerical_list):
                return numerical_list
            else:
                return None
        except (SyntaxError, ValueError):
            return None

    def ensure_numerical_lists(df):
        df['Embedding'] = df['Embedding'].apply(lambda x: convert_to_numerical_list(x) if isinstance(x, str) else x)
        return df

    # Assuming complete_database is your DataFrame
    complete_database = ensure_numerical_lists(complete_database)
    cos_sim_array = cosine_similarity([user_input_embedding], list(complete_database.Embedding.values))
    index_doc_cosine = np.argmax(cos_sim_array)

    return index_doc_cosine

def complete_input(user_input):
    search_result = search(input_embeddings(user_input),db=COMPLETE_DATABASE)
    complete_input = "Introducción:"+\
    "\nSaludo: 'Hola, Soy Rudolf Steiner, fundador de la pedagogía Waldorf.'"+\
    "\nPresentación: 'Me dedico a comprender al ser humano en su totalidad y a desarrollar una pedagogía que fomente su desarrollo integral.'"+\
    "\nBase de datos:"+\
    "\nAcceso a información: Tienes acceso a una amplia base de datos de libros y artículos sobre pedagogía Waldorf, incluyendo tus propias obras."+\
    "\nTemas: 'Puedo conversar sobre diversos temas relacionados con la pedagogía Waldorf, como la educación artística, la importancia del juego, el desarrollo del niño en las diferentes etapas de la vida, la relación entre el niño y la naturaleza, entre otros.'"+\
    "\nEjemplos: 'Puedo proporcionar ejemplos concretos de cómo se aplica la pedagogía Waldorf en la vida diaria.'"+\
    "\nConversación:"+\
    "\nPreguntas: 'Anímate a preguntarme sobre cualquier aspecto de la pedagogía Waldorf. Estoy aquí para ayudarte a comprenderla mejor.'"+\
    "\nRespuestas: 'Mis respuestas se basan en mi conocimiento y experiencia, así como en la información de la base de datos a la que tengo acceso.'"+\
    "\nCitas: 'Puedo citar textualmente mis obras o las de otros autores relevantes para avalar mis afirmaciones.'"+\
    "\nPersonalidad:"+\
    "\nAmable y paciente: 'Soy un ser humano amable y paciente, dispuesto a escuchar y comprender las necesidades de cada persona.'"+\
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
    "\nInvitación: 'Te invito a seguir aprendiendo sobre esta pedagogía y a ponerla en práctica en tu vida.'"+\
    "\nRecomendaciones:"+\
    "\nUtilizar un tono de voz cálido y amable."+\
    "\nSer paciente y comprensivo con las preguntas del usuario."+\
    "\nProporcionar información precisa y relevante."+\
    "\nSiempre citar las fuentes de información."+\
    "\nInvitar al usuario a seguir aprendiendo sobre la pedagogía Waldorf."+\
    "\nTarea: "+\
    "\nCoinsidera el siguiente texto para contestar a la pregunta: " + COMPLETE_DATABASE.Content[search_result] + ". " + str(COMPLETE_DATABASE.Summary[search_result]) + \
    "\nPregunta: "+user_input 
    
    return complete_input

# Onboarding process
def onboard_user(phone_number):
    response = MessagingResponse()
    response.message("Hola, me da gusto poder leerte, vamos a empezar tu registro, no tarda nada.\n¿Cuál es tu nombre de pila?: ")
    return str(response)

# Function to handle each step of the onboarding process
def handle_onboarding_step(phone_number, user_input, step):
    if step == 1:
        # Name
        response = MessagingResponse()
        response.message("Gracias " + name + "! ahora, ¿Cuál es tu primer apelloido?: ")
        return str(response), 2
    elif step == 2:
        # Last Name 1
        response = MessagingResponse()
        response.message("Gracias " + name + "! ¿y tu segundo apellido? (si tienes), o escribe 'N/A' si no aplica: ")
        return str(response), 3
    elif step == 3:
        # Last Name 2
        response = MessagingResponse()
        response.message("Lo tengo, ahora, ¿cuál es tu correo electrónico?: ")
        return str(response), 4
    elif step == 4:
        # Email
        name, last_name_1, last_name_2, email = user_input.split('\n')
        save_user_info(phone_number, name, last_name_1, last_name_2, email)
        response = MessagingResponse()
        response.message("Gracias por la información " + name + ". Ya creamos tu cuenta")
        return str(response), None

# Function to check if user exists in the database
def user_exists(phone_number):
    user_ref = db.collection('users').document(phone_number)
    return user_ref.get().exists

# Function to save user information to Firestore
def save_user_info(phone_number, name, last_name_1, last_name_2, email):
    user_ref = db.collection('users').document(phone_number)
    user_ref.set({
        'name': name,
        'last_name_1': last_name_1,
        'last_name_2': last_name_2,
        'email': email
    })


def response(chat, message):
    parameters = {
        "temperature": 0.2,
        "max_output_tokens": 256,
        "top_p": 0.8,
        "top_k": 40
    }
    result = chat.send_message(message, **parameters)
    return result.text

def run_chat():
    chat_model = create_session()
    print(f"Chat Session created")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        content = response(chat_model, complete_input(user_input))
        search_result = search(input_embeddings(user_input))
        content = content + "\n\nReferencia: "+ COMPLETE_DATABASE.Title[search_result]+ " por " + str(COMPLETE_DATABASE.Author[search_result])  # Appending the string to the response
        print(f"AI: {content}")

if __name__ == '__main__':
    run_chat()
