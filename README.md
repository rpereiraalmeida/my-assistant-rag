# **Projeto RAG Corporativo \- Arquitetura Local**

Documentação técnica da infraestrutura, pipeline de dados e mecanismos de segurança do sistema RAG local.

# **Objetivo**

Automatizar tarefas corporativas por meio do processamento e leitura de manuais internos, permitindo a geração precisa de relatórios operacionais via Inteligência Artificial Generativa.

# **A Arquitetura (Pipeline)**

* **LangChain:** Responsável pela etapa de ingestão e pré-processamento. Atua como o fatiador inteligente, dividindo os documentos em PDF em blocos menores (*chunks*) por meio do componente `RecursiveCharacterTextSplitter`.  
* **Google Gemini API:** Modelo de linguagem que atua como o cérebro terceirizado da solução. Executa a conversão de textos em embeddings vetoriais e a geração final das respostas, otimizando o consumo de recursos sem sobrecarregar a CPU (i7) e a memória RAM (8GB) do servidor.  
* **ChromaDB:** Banco de dados vetorial operando como memória persistente local diretamente no HD do Ubuntu. Armazena os embeddings gerados para evitar requisições repetitivas à API em nuvem e garantir baixa latência na recuperação.  
* **Streamlit:** Framework responsável pela interface de usuário interativa. Disponibiliza uma aplicação web de chat na porta `8501`, garantindo acesso via navegador web convencional e eliminando a necessidade de interação via terminal SSH.

# **Segurança Implementada**

* **Tailscale:** Configurado como VPN local criptografada *mesh*, permitindo o acesso seguro à aplicação sem a necessidade de expor portas da interface web diretamente para a internet pública.  
* **Autenticação:** Implementada na camada de aplicação através da biblioteca `streamlit-authenticator`, garantindo o controle de acesso por credenciais antes da liberação do chat.  
* **Auditoria de Logs:** Rastreabilidade mantida pelo arquivo `auditoria_rag.log`, registrando as interações e operações executadas pela equipe no sistema.

# **Hardwares Utilizados**

* **Sistema Operacional:** Servidor Ubuntu Server.  
* **Processador:** Intel Core i7-3537U (foco em eficiência energética, 2 núcleos / 4 threads).  
* **GPU:** NVIDIA Quadro K420 configurada com drivers *legacy* para eventuais rotinas secundárias; não utilizada no processamento ativo dos LLMs devido às restrições de memória VRAM.

