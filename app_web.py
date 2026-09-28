import os
import streamlit as st
import streamlit_authenticator as stauth
import logging
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from google import genai

# ==========================================
# 1. SETUP DE SEGURANÇA E LOGS
# ==========================================
logging.basicConfig(
    filename='auditoria_rag.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

st.set_page_config(page_title="RAG Corporativo", page_icon="🔒")

configuracao_usuarios = {
    'credentials': {
        'usernames': {
            'crodrigo': {
                'email': 'rodrigo.kpa02@gmail.com',
                'name': 'Rodrigo Almeida',
                'password': '$2b$12$6xR1AjrvIgMHpK0ZvzAOt.lfLfNrbBUtO1ozgRm8fdhTY56fTO7ey'
            }
        }
    },
    'cookie': {
        'expiry_days': 1, 
        'key': 'chave_criptografica_aleatoria_super_secreta',
        'name': 'cookie_sessao_rag'
    }
}

autenticador = stauth.Authenticate(
    credentials=configuracao_usuarios['credentials'],
    cookie_name=configuracao_usuarios['cookie']['name'],
    cookie_key=configuracao_usuarios['cookie']['key'],
    cookie_expiry_days=configuracao_usuarios['cookie']['expiry_days']
)

# ==========================================
# 2. INICIALIZAÇÃO DA IA (CACHEADA)
# ==========================================
# O @st.cache_resource impede que o servidor abra o banco de dados do zero a cada clique
@st.cache_resource
def iniciar_motor_ia():
    MINHA_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not MINHA_API_KEY:
        st.error("⚠️ Chave da API do Gemini não encontrada no servidor.")
        st.stop()
        
    google_client = genai.Client(api_key=MINHA_API_KEY)
    
    class NovoGoogleEmbeddingFunction(EmbeddingFunction):
        def __init__(self):
            pass

        def __call__(self, input: Documents) -> Embeddings:
            resposta = google_client.models.embed_content(
                model='gemini-embedding-001', 
                contents=input
            )
            return [vetor.values for vetor in resposta.embeddings]
            
    cliente_chroma = chromadb.PersistentClient(path="./meu_banco_vetorial_rag")
    colecao = cliente_chroma.get_or_create_collection(
        name="base_conhecimento", 
        embedding_function=NovoGoogleEmbeddingFunction()
    )
    
    return google_client, colecao

# ==========================================
# 3. INTERFACE E LÓGICA DO CHAT
# ==========================================
autenticador.login()

if st.session_state["authentication_status"]:
    # --- Usuário logado com sucesso ---
    autenticador.logout('Sair', 'sidebar')
    st.sidebar.write(f'Bem-vindo, **{st.session_state["name"]}**')
    
    st.title("🧠 Assistente de Manuais")
    
    # Carrega a IA e o Banco Vetorial
    google_client, colecao = iniciar_motor_ia()
    
    # Cria a memória de histórico de mensagens na tela
    if "mensagens" not in st.session_state:
        st.session_state.mensagens = []

    # Desenha as mensagens antigas no chat
    for msg in st.session_state.mensagens:
        st.chat_message(msg["role"]).markdown(msg["content"])
        
    # Caixa de texto para nova pergunta
    pergunta = st.chat_input("Digite sua dúvida sobre os manuais...")
    
    if pergunta:
        # 1. Registra no log de auditoria (Segurança)
        logging.info(f'Usuário: {st.session_state["name"]} | Pergunta: {pergunta}')
        
        # 2. Mostra a pergunta do usuário e salva no histórico
        st.chat_message("user").markdown(pergunta)
        st.session_state.mensagens.append({"role": "user", "content": pergunta})
        
        # 3. Busca no banco de dados e gera a resposta com o Gemini
        with st.chat_message("assistant"):
            with st.spinner("Buscando manuais no banco de dados..."):
                try:
                    resultados = colecao.query(query_texts=[pergunta], n_results=3)
                    
                    if not resultados['documents'][0]:
                        resposta_final = "Não encontrei informações sobre isso nos manuais cadastrados."
                    else:
                        contexto = "\n---\n".join(resultados['documents'][0])
                        prompt_enriquecido = f"""
                        Responda à pergunta do usuário baseando-se EXCLUSIVAMENTE neste contexto:
                        {contexto}
                        
                        Pergunta: {pergunta}
                        """
                        
                        resposta_gemini = google_client.models.generate_content(
                            model='gemini-flash-lite-latest',
                            contents=prompt_enriquecido
                        )
                        resposta_final = resposta_gemini.text
                        
                    # Mostra a resposta na tela
                    st.markdown(resposta_final)
                    # Salva a resposta no histórico da tela
                    st.session_state.mensagens.append({"role": "assistant", "content": resposta_final})
                    
                except Exception as e:
                    st.error(f"Erro ao processar a resposta: {e}")

elif st.session_state["authentication_status"] is False:
    st.error('Usuário ou senha incorretos.')
    logging.warning('Tentativa de login falha registrada.')
    
elif st.session_state["authentication_status"] is None:
    st.warning('Por favor, insira suas credenciais de acesso corporativo.')
