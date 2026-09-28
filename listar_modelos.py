import google.generativeai as genai
import os

# Certifique-se de que sua variável de ambiente com a chave da API está configurada
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

print("Modelos disponíveis para geração de texto:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)
