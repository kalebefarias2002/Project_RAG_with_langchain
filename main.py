from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from typer import prompt


caminho_db = "db"

prompt_template = """Você é um assistente de IA que responde perguntas com base em informações extraídas de documentos.
{pergunta}

com base nessas informações:
{base_de_conhecimento}
se você não souber a resposta, diga "Desculpe, não sei a resposta para essa pergunta."."""


def perguntar():
    pergunta = input("Digite sua pergunta: ")

    #carregar o banco de dados

    funcao_embedding = OpenAIEmbeddings()
    db = Chroma(persist_directory=caminho_db, embedding_function=funcao_embedding)

    #compara a pergunta com o banco de dados e retorna os resultados mais relevantes
    resultado = db._similarity_search_with_relevance_scores(pergunta, k=3)
    if len(resultado) == 0 or resultado[0][1] < 0.5:
        print("Desculpe, não sei a resposta para essa pergunta.")
        return

    textos_resultado = []
    for resultado in resultado:
        texto = resultado.page_content
        textos_resultado.append(texto)

    base_conhecimento = "\n\n----\n\n".join(textos_resultado)

    modelo = OpenAIEmbeddings()
    texto_resposta = modelo.invoke(prompt)
    print(f"Resposta: {texto_resposta}")

perguntar()

    

