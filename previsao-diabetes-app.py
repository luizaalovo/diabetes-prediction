import streamlit as st
import pickle
import numpy as np
import pandas as pd

from PIL import Image

#Definição de variáveis
path = 'models/'
#Carregando modelos
model = pickle.load(open(path+"modelo.pkl", 'rb'))
scaler = pickle.load(open(path+"standard_scaler.pkl", 'rb'))

#Funções
def definiGender(gender):
  '''
  Função que realiza um OneHotEncoder/Get Dummies manual a partir da variável 'gender'.
  '''
  if gender == 'Female' or 'Feminino':
    return {'gender_Female':1, 'gender_Male':0, 'gender_Other':0}
  if gender == 'Male' or 'Masculino':
    return {'gender_Female':0, 'gender_Male':1, 'gender_Other':0}
  if gender == 'Male' or 'Other':
    return {'gender_Female':0, 'gender_Male':0, 'gender_Other':1}

def definirSmokingHistory(smokinghistory):
  '''
  Função que realiza um OneHotEncoder/Get Dummies manual a partir da variável 'smoking_history'.
  '''
  classification_smoking = {'Sem Informação':'No Info', 'Nunca':'never', 'Não Atualmente':'not current', 'Antes':'former', 'Atualmente':'current', 'Sempre':'ever'}

  smoking_history = {
                    'never': {'smoking_history_No Info':0, 'smoking_history_current':0, 'smoking_history_ever':0,
                              'smoking_history_former':0, 'smoking_history_never':1, 'smoking_history_not current':0},
                    'No Info': {'smoking_history_No Info':1, 'smoking_history_current':0, 'smoking_history_ever':0,
                                'smoking_history_former':0, 'smoking_history_never':0, 'smoking_history_not current':0},
                    'current': {'smoking_history_No Info':0, 'smoking_history_current':1, 'smoking_history_ever':0,
                                'smoking_history_former':0, 'smoking_history_never':0, 'smoking_history_not current':0},
                    'former': {'smoking_history_No Info':0, 'smoking_history_current':0, 'smoking_history_ever':0,
                                'smoking_history_former':1, 'smoking_history_never':0, 'smoking_history_not current':0},
                    'ever': {'smoking_history_No Info':0, 'smoking_history_current':0, 'smoking_history_ever':1,
                              'smoking_history_former':0, 'smoking_history_never':0, 'smoking_history_not current':0},
                    'not current': {'smoking_history_No Info':0, 'smoking_history_current':0, 'smoking_history_ever':0,
                                    'smoking_history_former':0, 'smoking_history_never':0, 'smoking_history_not current':1}
                    }

  return smoking_history[classification_smoking[smokinghistory]]

def definirHipertensao(hipertensao):
  '''
  Função que realiza um OneHotEncoder/Get Dummies manual a partir da variável 'hypertension'.
  '''
  classification_hypertension = {'Sim':1, 'Não':0}
  return classification_hypertension[hipertensao]

def definirDoencaCardiaca(doenca_cardiaca):
  '''
  Função que realiza um OneHotEncoder/Get Dummies manual a partir da variável 'heart_disease'.
  '''
  classification_heart_disease = {'Sim':1, 'Não':0}
  return classification_heart_disease[doenca_cardiaca]

##Início App
#Configuração App
st.set_page_config(page_icon="🩸", page_title="Diabetes", layout="centered")

#Menu - side bar
st.sidebar.title("Menu")
st.sidebar.write("Os dados estão disponíveis no [Kaggle]('https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset'), e demonstra informações médicas e demográficas de pacientes.")
st.sidebar.write("---")
st.sidebar.write("Contate o autor pelo QRCode")
st.sidebar.image(image=Image.open("./assets/qrcode_linkedin.png"), caption="LinkedIn Luíza A. Lovo", width=300)
scol1, scol2 = st.sidebar.columns(2)
scol1.write("[![GitHub](https://img.shields.io/badge/-GitHub-333333?style=for-the-badge&logo=github)](https://github.com/luizaalovo)")
scol2.write("[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/luiza-angelo-lovo)")
st.sidebar.write("---")
st.sidebar.write('Por Luíza Angelo Lovo')

#Página principal
st.title("Previsão de Diabetes com o uso de Machine Learning 🩸")
st.write("Classificação com base em dados do histórico médico e informações demográficas de pacientes.")

#Formulário
with st.form(key="my_form"):
    genero_paciente=st.selectbox("Qual o seu gênero?", ["Feminino", "Masculino", "Outros"], index=0, help="Opção na qual se segmenta o paciente pelo gênero.")
    idade_paciente = st.slider("Qual a sua idade:", min_value=0, max_value=100)
  
    col1, col2, col3 = st.columns(3)
    hipertensao_paciente = col1.radio("Tem Hipertensão?", ["Sim", "Não"])
    doenca_cardiaca_paciente = col2.radio("Tem Doença Cardíaca?", ["Sim", "Não"])
    historico_tabagismo_paciente = col3.selectbox("Qual o seu histórico de tabagismo:", ["Sem Informação", "Nunca", "Não Atualmente", "Antes", "Atualmente", "Sempre"], index=0, help="Opção na qual mostra um dos dados demográficos do paciente.")
  
    imc_paciente = st.number_input("Qual o seu IMC:", min_value=0.00, max_value=100.00, help="IMC = Peso / Altura*Altura")
    
    col1, col2 = st.columns(2)
    hemoglobina_paciente = col1.number_input("Qual o seu nível de Hemoglobina Glicada - A1c:", min_value=0.00, max_value=20.00)
    glicose_paciente = col2.number_input("Qual o seu nível de glicose:", min_value=0, max_value=600)
    
    submit_button = st.form_submit_button(label="Classificar 🎉")
    
#Dicionário para os dados do paciente
dict_dados = {}

#Insere dados gerais do paciente
dict_dados.update({'age': idade_paciente, 'hypertension': definirHipertensao(hipertensao_paciente), 
                  'heart_disease': definirDoencaCardiaca(doenca_cardiaca_paciente),
                  'bmi': imc_paciente, 'HbA1c_level': hemoglobina_paciente, 'blood_glucose_level': glicose_paciente})

#Insere dados do 'gênero' do paciente já com o getdummies realizado.
dict_dados.update(definiGender(genero_paciente))

#Insere dados do 'histórico de tabagismo' do paciente já com o getdummies realizado.
dict_dados.update(definirSmokingHistory(historico_tabagismo_paciente))

st.write('---')

#Classificador de Diabetes a partir dos dados do paciente.
if submit_button:
  with st.spinner('Classificando...'):
    st.info(body="""Paciente:""", icon="ℹ️")
    with st.expander(label="""📋 Dados do paciente 📋"""):
      st.json(dict_dados)

    st.write('---')
    
    #Previsão do modelo
    st.success("Previsão: ", icon="✅")
    previsao_modelo = model.predict(scaler.transform(np.array(list(dict_dados.values())).reshape(1, -1)))
    previsao_proba_modelo = model.predict_proba(scaler.transform(np.array(list(dict_dados.values())).reshape(1, -1)))
    df_previsao = pd.DataFrame(data=previsao_proba_modelo*100, columns=["Negativo", "Positivo"])
    df_previsao['Negativo'] = df_previsao['Negativo'].map('{:.2f}%'.format)
    df_previsao['Positivo'] = df_previsao['Positivo'].map('{:.2f}%'.format)
    df_previsao = df_previsao.rename(index={0: "Resultado"})

    if previsao_modelo == 1:
      st.write(f"Positivo para diabetes em \n{df_previsao['Positivo'][0]}")
      st.write(df_previsao)
      st.balloons()
    else:
      st.write(f"Negativo para diabetes em \n{df_previsao['Negativo'][0]}")
      st.write(df_previsao)
      st.balloons()

    st.write('---')

else:
  st.warning(body="Insira os dados do paciente!!!", icon="⚠️")