import os
import time
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from google import genai
from google.genai import errors
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================
# 1. SETUP DA API E BANCO VETORIAL
# ==========================================
MINHA_API_KEY = os.environ.get("GEMINI_API_KEY")
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

funcao_embedding = NovoGoogleEmbeddingFunction()
cliente_chroma = chromadb.PersistentClient(path="./meu_banco_vetorial_rag")
colecao = cliente_chroma.get_or_create_collection(
    name="base_conhecimento", 
    embedding_function=funcao_embedding
)

# ==========================================
# 2. LEITURA E CHUNKING INTELIGENTE (PEDAÇO POR PEDAÇO)
# ==========================================
pasta_docs = "./meus_documentos"
documentos_novos = []
ids_novos = []

# Busca os IDs exatos que já existem no banco
ids_ja_processados = set(colecao.get()['ids'])

fatiador = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""]
)

print(f"Lendo e fatiando arquivos da pasta '{pasta_docs}'...")

if os.path.exists(pasta_docs):
    for arquivo in os.listdir(pasta_docs):
        caminho_completo = os.path.join(pasta_docs, arquivo)
        texto_completo = ""
        
        # Leitura dos formatos suportados
        if arquivo.endswith(".txt"):
            with open(caminho_completo, "r", encoding="utf-8") as f:
                texto_completo = f.read()
        elif arquivo.endswith(".pdf"):
            leitor = PdfReader(caminho_completo)
            for pagina in leitor.pages:
                texto = pagina.extract_text()
                if texto:
                    texto_completo += texto + "\n"

        # Fatiamento e verificação individual de cada pedaço
        if texto_completo.strip():
            pedacos = fatiador.split_text(texto_completo)
            chunks_novos_deste_arquivo = 0
            
            for i, pedaco in enumerate(pedacos):
                id_do_chunk = f"{arquivo}_chunk_{i}"
                
                # Adiciona na fila apenas se este pedaço específico não estiver no banco
                if id_do_chunk not in ids_ja_processados:
                    documentos_novos.append(pedaco)
                    ids_novos.append(id_do_chunk)
                    chunks_novos_deste_arquivo += 1
            
            if chunks_novos_deste_arquivo > 0:
                print(f"📄 '{arquivo}': {chunks_novos_deste_arquivo} pedaços novos encontrados para envio.")
            else:
                print(f"⏭️  '{arquivo}' já está 100% vetorizado. Pulando.")
else:
    print(f"⚠️ A pasta '{pasta_docs}' não foi encontrada.")

# ==========================================
# 3. ATUALIZAÇÃO DO BANCO EM LOTES (COM RETRY E PULO)
# ==========================================
if documentos_novos:
    print(f"\nVetorizando {len(documentos_novos)} pedaços (chunks) pendentes. Aguarde...")
    
    tamanho_lote = 90 
    cancelar_envio = False # Flag para rastrear se o usuário quer pular
    
    for i in range(0, len(documentos_novos), tamanho_lote):
        if cancelar_envio:
            break # Sai do loop de envio se o usuário apertou CTRL+C
            
        lote_docs = documentos_novos[i : i + tamanho_lote]
        lote_ids = ids_novos[i : i + tamanho_lote]
        
        print(f"-> Enviando lote do item {i+1} ao {i + len(lote_docs)} de {len(documentos_novos)}...")
        
        while True:
            try:
                colecao.upsert(documents=lote_docs, ids=lote_ids)
                break 
            except Exception as e:
                if "429" in str(e):
                    print("⏳ Cota da API atingida! O servidor vai aguardar 60 segundos...")
                    print("💡 DICA: Pressione 'CTRL + C' agora se quiser parar de enviar e ir direto para o chat!")
                    
                    try:
                        time.sleep(60)
                        print("▶️ Retomando envio do lote...")
                    except KeyboardInterrupt:
                        # Se você apertar CTRL+C durante o tempo de espera, ele cai aqui!
                        print("\n⏭️ Envio pausado pelo usuário! Vamos para o chat com o que já foi salvo.")
                        cancelar_envio = True
                        break # Quebra o 'while' interno
                else:
                    print(f"⚠️ Erro inesperado ao salvar lote: {e}")
                    cancelar_envio = True
                    break
        
    print("✅ Processo de atualização finalizado!\n")
else:
    print("\n✅ Todos os arquivos da pasta já estão sincronizados no banco.\n")

# ==========================================
# 4. CHAT EM TEMPO REAL VIA TERMINAL
# ==========================================
print("="*40)
print("🤖 CHATBOT RAG ATIVADO")
print("Digite 'sair' a qualquer momento para encerrar.")
print("="*40 + "\n")

while True:
    pergunta_usuario = input("👤 Você: ")
    
    if pergunta_usuario.lower() in ['sair', 'exit', 'quit']:
        print("Encerrando o sistema. Até logo!")
        break
        
    if not pergunta_usuario.strip():
        continue

    # Busca no banco vetorial pelos 3 pedaços mais parecidos
    resultados = colecao.query(query_texts=[pergunta_usuario], n_results=3)
    
    if not resultados['documents'][0]:
        print("🤖 Gemini: Não encontrei nada sobre isso nos meus documentos.\n")
        continue
        
    contexto_recuperado = "\n---\n".join(resultados['documents'][0])

    prompt_enriquecido = f"""
    Responda à pergunta do usuário baseando-se EXCLUSIVAMENTE neste contexto:
    {contexto_recuperado}
    
    Pergunta: {pergunta_usuario}
    """

    try:
        resposta_final = google_client.models.generate_content(
            model='gemini-flash-lite-latest',
            contents=prompt_enriquecido
        )
        print(f"🤖 Gemini: {resposta_final.text}\n")
        
    except errors.ClientError as e:
        if "429" in str(e):
            print("⏳ Gemini: Limite da cota gratuita atingido. Aguarde um minuto.\n")
        else:
            print(f"⚠️ Erro de API: {e}\n")
    except Exception as e:
        print(f"⚠️ Erro inesperado: {e}\n")
