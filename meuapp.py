import streamlit as st
import pandas as pd
import requests
import json
import time
import numpy as np
from datetime import datetime
from typing import List

from sklearn.metrics.pairwise import cosine_similarity
from google.genai import types
from google import genai 

from pydantic import BaseModel, Field


# --- Configuração das Chaves de API ---
JINA_API_KEY = st.secrets['JINA_API_KEY']
GEMINI_API_KEY = st.secrets['GEMINI_API_KEY']

def buscar_google_news(termo):
    from GoogleNews import GoogleNews

    # Inicializa o objeto GoogleNews com os parâmetros desejados
    googlenews = GoogleNews(
        lang='pt-BR',        # Define o idioma para português do Brasil
        period='1d',         # Define o período para os últimos 7 dias
        encode='utf-8'       # Define a codificação para UTF-8
    )

    # Realiza a busca por notícias relacionadas ao termo 'tecnologia'
    googlenews.search(termo)

    # Define o número máximo de resultados desejados
    max_resultados = 1000
    resultados = []
    pagina = 1

    # Itera sobre as páginas de resultados até atingir o número desejado
    while len(resultados) < max_resultados:
        googlenews.get_page(pagina)
        noticias = googlenews.result()
        if not noticias:
            break  # Encerra se não houver mais resultados
        resultados.extend(noticias)
        pagina += 1

    # Limita a lista de resultados ao número máximo desejado
    resultados = resultados[:max_resultados]

    # Separandos as noticias
    links_noticias = [noticia['link'].split('&ved')[0] for noticia in resultados]

    # Exibe os resultados
    quantidade_noticias = len(resultados)
    print(f'Quantidade de notícias retornadas do GoogleNews: {quantidade_noticias}')

    # Coloca todas as noticias num dataframe
    import pandas as pd
    df = pd.DataFrame(resultados)
    df['link'] = df['link'].str.split('&ved').str[0]
    # a coluna media deve ser renomeada para source
    df.rename(columns={'media': 'source'}, inplace=True)
    # dropar datetime e img
    df.drop(columns=['datetime', 'img'], inplace=True)

    return df



def pega_noticias(INTERESSE):
    """
    Usa o interesse do usuário, utiliza o Gemini para extrair palavras-chave e gerar temas de busca,
    pesquisa no Google News para cada tema e retorna um DataFrame combinado e limpo.
    """
    # ADICIONADO: A biblioteca 'google.genai' é importada dentro da função, conforme o código original.
    from google import genai
    # ADICIONADO: A biblioteca 'pandas' é importada como 'pd', conforme o código original.
    import pandas as pd

    # ADICIONADO: O cliente Gemini é inicializado aqui para a geração de temas.
    client = genai.Client(api_key = GEMINI_API_KEY)

    # ADICIONADO: O prompt para o Gemini gerar os temas de busca com base no interesse do usuário.
    prompt = f"""
    Dado o seguinte interesse do usuário, extraia palavras-chave relevantes e gere 3 a 5 temas de busca relacionados que podem ser usados para encontrar notícias no Google News.
    Formato de saída: Uma lista de strings, onde cada string é um tema de busca.

    Interesse do usuário: {INTERESSE}
    """

    # ADICIONADO: Bloco try-except para a chamada da API Gemini e geração de temas.
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash", # MANTIDO: O modelo do Gemini não foi alterado, conforme solicitado.
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": {"type": "array", "items": {"type": "string"}},
            },
        )
        search_themes = json.loads(response.text)
        print("Temas de busca gerados por Gemini:", search_themes)
    except Exception as e:
        print(f"Erro ao gerar temas de busca com Gemini: {e}")
        return pd.DataFrame() # Retorna um DataFrame vazio em caso de falha.

    # ADICIONADO: Inicialização de um DataFrame vazio para armazenar todas as notícias.
    all_news_df = pd.DataFrame()

    # ADICIONADO: Loop que itera sobre os temas gerados e busca notícias para cada um.
    for theme in search_themes:
        print(f"Buscando notícias para o tema: {theme}")
        # A função 'buscar_google_news' existente no CÓDIGO 2 é chamada aqui.
        news_df = buscar_google_news(theme)
        if not news_df.empty:
            news_df['search_theme'] = theme
            all_news_df = pd.concat([all_news_df, news_df], ignore_index=True)

    # ADICIONADO: Bloco para limpar e remover duplicatas do DataFrame combinado.
    if not all_news_df.empty:
        all_news_df.dropna(subset=['link'], inplace=True)
        all_news_df = all_news_df.drop_duplicates(subset=['link'], keep='first')
        all_news_df = all_news_df.drop_duplicates(subset=['title'], keep='first')
        all_news_df.reset_index(drop=True, inplace=True)

    # ADICIONADO: Mensagem final informando o total de notícias únicas encontradas.
    print(f"Busca combinada concluída! {all_news_df.shape[0]} notícias únicas encontradas.")
    # ADICIONADO: Retorno do DataFrame final com todas as notícias.
    return all_news_df

def ordenar_noticias_por_similaridade(interesse, df_noticias, top_n=10):

    TEXTOS = df_noticias['title'].to_list()

    client = genai.Client(api_key = st.secrets['GEMINI_API_KEY'])

    result = client.models.embed_content(
                model="gemini-embedding-001",
                contents=interesse)

    interesse_embed  = np.array(result.embeddings[0].values)

    # Processar os textos em lotes de 100
    VETORES = []
    for i in range(0, len(TEXTOS), 100):
        batch_textos = TEXTOS[i:i+100]
        embeddings_result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=batch_textos,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        ).embeddings
        VETORES.extend([np.array(e.values) for e in embeddings_result])

    # Certifique-se de que interesse_embed é um array numpy 2D (necessário para cosine_similarity)
    interesse_embed_2d = interesse_embed.reshape(1, -1)

    # Calcular a similaridade de cosseno entre o vetor de interesse e cada vetor na lista VETORES
    similaridades = [cosine_similarity(interesse_embed_2d, v.reshape(1, -1))[0][0] for v in VETORES]

    # Adicionar as similaridades ao DataFrame
    df_noticias['score'] = similaridades

    # Ordenar o DataFrame pelo score
    df_noticias.sort_values(by='score', ascending=False, inplace = True)

    # Retornar apenas as top_n notícias e resetar o index
    return df_noticias.head(top_n).reset_index(drop=True)

def extrair_conteudo_noticias(df_noticias):
    """Extrai o conteúdo completo dos artigos usando a Jina AI API."""
    headers = {
        "Authorization": f"Bearer {st.secrets['JINA_API_KEY']}",
        "X-Engine": "browser",
        "X-Return-Format": "markdown"
    }

    total_noticias = len(df_noticias)
    # Removed Streamlit progress bar and status text

    conteudos = []
    for index, row in df_noticias.iterrows():
        print(f"Extraindo notícia {index + 1}/{total_noticias}: {row['title'][:50]}...") # Changed status_text.text to print
        url = f"https://r.jina.ai/{row['link']}"
        while True:
            try:
                response = requests.get(url, headers=headers, timeout=90)
                response.raise_for_status()
                conteudos.append(response.text)
                break
            except requests.exceptions.RequestException as e:
                conteudos.append(f"Erro ao buscar conteúdo para o título '{row['title']}': {e}")


    df_noticias['content'] = conteudos
    return df_noticias


from google import genai
from pydantic import BaseModel, Field
import time
import json

def processa_noticias_com_gemini(articles_df):
    client = genai.Client(api_key = GEMINI_API_KEY)

    class Noticia(BaseModel):
        titulo: str = Field(..., description="O título da notícia.")
        data_de_publicacao: str = Field(..., description="A data em que a notícia foi publicada. Use sempre o formato: 'DD/MM/AAAA'.")
        autor: str = Field(..., description="O nome do autor da notícia.")
        portal: str = Field(..., description="O nome do portal de notícias onde a notícia foi publicada.")
        resumo_curto: str = Field(..., description="Um resumo conciso da notícia em torno de 50 palavras. De preferência para colocar informação adicional ao titulo (nao repetir a informacao do titulo)")
        resumo_maior: str = Field(..., description="Um resumo mais detalhado da notícia em torno de 500 palavras.")
        pontos_principais: List[str] = Field(..., description="um resumo da noticia em formato de lista item a item")
        noticia_completa: str = Field(..., description="O texto completo da notícia.")
        links_de_imagens: List[str] = Field(..., description="Uma lista de URLs das imagens associadas à notícia. Considere apenas aquelas relevantes para a noticia. Descarte logos, divulgacoes, etc...")
        tags_relevantes: List[str] = Field(..., description="Uma lista de tags ou palavras-chave relevantes para a notícia.")
        prompt_satira_imagem: str = Field(..., description="Um prompt de sátira, baseado no conteúdo da notícia, para ser usado em um gerador de imagens. Deve ser criativo e com um tom humorístico ou irônico.")


    respostas = []
    for texto in articles_df['content']:
        print(f"Fazendo extração do {texto[:40]}...")
        while True:
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents = f"Extraia informacoes da noticia em texto cru dada a seguir: \n\n {texto}",
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": Noticia,
                    },
                )
                break
            except Exception as e:
                print(f"Erro na API: {e} \nTentando novamente em 3s...")
                time.sleep(3)




        # Use the response as a JSON string.
        respostas.append(response.text)



    # Converter cada string JSON em um dicionário
    lista_de_dicionarios = [json.loads(json_string) for json_string in respostas]

    processados_df = pd.DataFrame(lista_de_dicionarios)
    return processados_df


def gerar_card_noticia(noticia: dict, idx: int) -> str:
    """Gera HTML para um card de notícia a partir de um dicionário (linha do dataframe 'processados')."""

    # Extrai dados da notícia usando as chaves do dicionário (nomes das colunas do dataframe 'processados')
    titulo = noticia.get('titulo', '')
    portal = noticia.get('portal', '')
    data_pub = noticia.get('data_de_publicacao', '') # Usando 'data_de_publicacao'
    resumo_breve = noticia.get('resumo_curto', '')  # Usando 'resumo_curto'
    resumo_expandido = noticia.get('resumo_maior', '') # Usando 'resumo_maior'
    tags = noticia.get('tags_relevantes', []) # Usando 'tags_relevantes'
    url_original = noticia.get('link', '') # Usando 'link'
    caminho_imagem = noticia.get('links_de_imagens', []) # Usando 'links_de_imagens' - pode ser uma lista
    prompt_satira_imagem = noticia.get('prompt_satira_imagem', '') # Usando 'prompt_satira_imagem'
    pontos_principais = noticia.get('pontos_principais', []) # Usando 'pontos_principais'

    # Pega o primeiro link de imagem, se existir, caso contrário, usa uma string vazia
    imagem_url = caminho_imagem[0] if caminho_imagem else ''

    # Converte tags para string
    tags_str = ', '.join(tags) if tags else ''

    # Converte pontos principais para HTML list items
    pontos_principais_html = "".join([f"<li>{p}</li>" for p in pontos_principais]) if pontos_principais else ""


    # Gera HTML do card
    card_html = f"""
    <div class="card-noticia">
        <div class="card-header">
            <h3 class="card-titulo">{titulo}</h3>
            <div class="card-meta">
                <span class="portal">{portal}</span>
                <span class="data">{data_pub}</span>
            </div>
        </div>

        <div class="card-content">
            <div class="card-imagem">
                <img src="{imagem_url}" alt="{titulo}" onerror="this.style.display='none'">
            </div>

            <div class="card-texto">
                <p class="resumo-breve"><strong>Resumo Curto:</strong> {resumo_breve}</p>

                <div class="button-array">
                    <button class="popover-button" data-popover-target="#popover-resumo-{idx}">Resumo Completo</button>
                    <button class="popover-button" data-popover-target="#popover-tags-{idx}">Tags Relevantes</button>
                    <button class="popover-button" data-popover-target="#popover-satira-{idx}">Prompt de Sátira</button>
                    <button class="popover-button" data-popover-target="#popover-pontos-{idx}">Pontos Principais</button>
                </div>

                <a href="{url_original}" target="_blank" class="btn-ler-mais">
                    Ler notícia completa
                </a>
            </div>
        </div>

        <!-- Popover content -->
        <div id="popover-resumo-{idx}" class="popover-content" style="display: none;">
            <div class="popover-header">
                <h4>Resumo Completo</h4>
                 <span class="close-popover">&times;</span>
            </div>
            <div class="popover-body">
                <p>{resumo_expandido}</p>
            </div>
        </div>

        <div id="popover-tags-{idx}" class="popover-content" style="display: none;">
             <div class="popover-header">
                <h4>Tags Relevantes</h4>
                 <span class="close-popover">&times;</span>
            </div>
             <div class="popover-body">
                 <div>{tags_str}</div>
            </div>
        </div>

        <div id="popover-satira-{idx}" class="popover-content" style="display: none;">
            <div class="popover-header">
                <h4>Prompt de Sátira</h4>
                <span class="close-popover">&times;</span>
            </div>
             <div class="popover-body">
                 <p>{prompt_satira_imagem}</p>
            </div>
        </div>

        <div id="popover-pontos-{idx}" class="popover-content" style="display: none;">
            <div class="popover-header">
                <h4>Pontos Principais</h4>
                <span class="close-popover">&times;</span>
            </div>
             <div class="popover-body">
                 <ul>{pontos_principais_html}</ul>
            </div>
        </div>
    </div>
    """

    return card_html

def gerar_html_newsletter(df: pd.DataFrame, interesse: str) -> str:
    """Gera um arquivo HTML para a newsletter a partir do DataFrame processado."""

    data_atual = datetime.now().strftime("%d/%m/%Y")

    # Início do HTML da newsletter
    html_content = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Newsletter sobre {interesse}</title>
        <style>
            body {{ font-family: sans-serif; line-height: 1.6; margin: 20px; background-color: #e0e0e0;}} /* Added background color to body */
            .container {{ max-width: 800px; margin: auto; background: #f9f9f9; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1); }} /* Added box-shadow to container */
            .header {{ text-align: center; margin-bottom: 30px; }}
            .header h1 {{ color: #333; }}
            .header p {{ color: #666; }}
            .card-noticia {{ border: 1px solid #ddd; margin-bottom: 20px; border-radius: 8px; overflow: hidden; background: #d3dbde; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1); position: relative;}} /* Added box-shadow to card and changed background color, added position: relative for popover positioning */
            .card-header {{ background: #d3dbde; padding: 15px; border-bottom: 1px solid #ddd; }} /* Changed background color to match card */
            .card-titulo {{ margin: 0; color: #007bff; }}
            .card-meta {{ font-size: 0.9em; color: #555; margin-top: 5px; }}
            .card-meta span {{ margin-right: 15px; }}
            .card-content {{ display: flex; flex-wrap: wrap; padding: 15px; }}
            .card-imagem {{ flex: 1 1 150px; margin-right: 15px; }}
            .card-imagem img {{ max-width: 100%; height: auto; border-radius: 4px; }}
            .card-texto {{ flex: 2 1 300px; }}
            .resumo-breve {{ font-weight: normal; margin-bottom: 10px; }}
            .button-array {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; margin-bottom: 10px; }} /* Grid for button array */
            .popover-button {{
                padding: 10px;
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 0.9em;
                text-align: center;
            }}
            .popover-button:hover {{
                background-color: #0056b3;
            }}
            .btn-ler-mais {{
                display: block; /* Make it a block element to take full width */
                width: 100%; /* Make it wider */
                background-color: #28a745; /* Elegant green color */
                color: white;
                padding: 12px 15px; /* Slightly more padding */
                text-decoration: none;
                border-radius: 5px;
                margin-top: 15px; /* Add some space above the button */
                text-align: center; /* Center the text */
                font-size: 1.1em; /* Slightly larger font */
                transition: background-color 0.3s ease; /* Smooth hover effect */
            }}
            .btn-ler-mais:hover {{
                background-color: #218838; /* Darker green on hover */
            }}
            .footer {{ text-align: center; margin-top: 30px; font-size: 0.8em; color: #777; }}

            /* Popover styles */
            .popover-content {{
                display: none; /* Hidden by default */
                position: absolute;
                background-color: #f9f9f9;
                box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
                padding: 15px;
                z-index: 1;
                border-radius: 8px;
                max-width: 500px; /* Increased max-width */
                word-wrap: break-word; /* Break long words */
                top: 50%; /* Position relative to the card */
                left: 50%;
                transform: translate(-50%, -50%); /* Center the popover */
                border: 1px solid #ddd;
            }}

            .popover-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid #ddd;
                padding-bottom: 10px;
                margin-bottom: 10px;
            }}

            .popover-header h4 {{
                margin: 0;
                color: #333;
            }}

            .popover-body {{
                /* Add styles for the body if needed */
            }}

            .close-popover {{
                color: #aaa;
                font-size: 20px;
                font-weight: bold;
                cursor: pointer;
            }}

            .close-popover:hover,
            .close-popover:focus {{
                color: #000;
                text-decoration: none;
            }}

            .popover-content ul {{
                list-style-type: disc;
                margin-left: 20px;
            }}
            .popover-content li {{
                margin-bottom: 5px;
            }}


        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Newsletter sobre {interesse}</h1>
            </div>
            """

    # Adiciona cada card de notícia
    for idx, row in df.iterrows():
        noticia_dict = row.to_dict()
        html_content += gerar_card_noticia(noticia_dict, idx)

    # Fim do HTML da newsletter
    html_content += """
        </div>

        <script>
            document.addEventListener('DOMContentLoaded', function() {
                const buttons = document.querySelectorAll('.popover-button');
                const closeButtons = document.querySelectorAll('.close-popover');

                buttons.forEach(button => {
                    button.addEventListener('click', function() {
                        const target = document.querySelector(this.dataset.popoverTarget);
                        // Hide all other popovers
                        document.querySelectorAll('.popover-content').forEach(popover => {
                            if (popover !== target) {
                                popover.style.display = 'none';
                            }
                        });
                        // Toggle the display of the target popover
                        target.style.display = target.style.display === 'none' ? 'block' : 'none';
                    });
                });

                closeButtons.forEach(button => {
                    button.addEventListener('click', function() {
                        this.closest('.popover-content').style.display = 'none';
                    });
                });

                // Hide popover when clicking outside
                document.addEventListener('click', function(event) {
                    if (!event.target.classList.contains('popover-button') && !event.target.closest('.popover-content')) {
                        document.querySelectorAll('.popover-content').forEach(popover => {
                            popover.style.display = 'none';
                        });
                    }
                });
            });
        </script>
    </body>
    </html>
    """

    with open("newsletter.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print("Newsletter HTML gerada e salva como 'newsletter.html'")

    return html_content


import streamlit as st
import pandas as pd # Adicione esta linha no topo do seu arquivo se ainda não estiver lá

# --- INTERFACE DO USUÁRIO (STREAMLIT) ---

st.title('Minha Newsletter Personalizada')

# 1. ENTRADA DE DADOS DO USUÁRIO
# Apenas os campos de interesse e número de notícias são necessários agora.
INTERESSE = st.text_input('Qual é o seu interesse específico? (Ex: o futuro da inteligência artificial no Brasil)')
TOP_NOTICIAS = st.number_input('Número de notícias desejado', min_value=1, max_value=20, value=5)

# 2. BOTÃO PARA INICIAR O PROCESSO
if st.button('Gerar Newsletter'):

    # 3. VALIDAÇÃO DO INPUT
    # Verificamos apenas se o INTERESSE foi preenchido.
    if not INTERESSE:
        st.warning('Por favor, preencha o seu interesse específico para continuar.')
    else:
        # 4. EXECUÇÃO DAS ETAPAS (Tudo acontece dentro deste 'else')
        
        # Etapa 1: Pega as notícias
        with st.spinner('Buscando e gerando temas de notícias... 📰'):
            pegas = pega_noticias(INTERESSE)

        # Verifica se alguma notícia foi encontrada antes de prosseguir
        if pegas.empty:
            st.error('Nenhuma notícia encontrada para os temas gerados. Tente um interesse diferente.')
        else:
            # Etapa 2: Ordena por similaridade
            with st.spinner('Analisando e ordenando por relevância... 🧠'):
                top_noticias = ordenar_noticias_por_similaridade(
                    interesse=INTERESSE,
                    df_noticias=pegas,
                    top_n=int(TOP_NOTICIAS)
                )

            # Etapa 3: Extrai o conteúdo completo
            with st.spinner('Extraindo o conteúdo completo dos artigos... 📄'):
                extracoes = extrair_conteudo_noticias(top_noticias)

            # Etapa 4: Processa com a IA
            with st.spinner('Criando resumos e insights com IA... ✨'):
                processados = processa_noticias_com_gemini(extracoes)

            # Etapa 5: Gera o HTML final
            with st.spinner('Montando sua newsletter personalizada... HTML'):
                final = pd.concat([extracoes.reset_index(drop=True), processados.reset_index(drop=True)], axis=1)
                newsletter_html = gerar_html_newsletter(final, INTERESSE)

            # 5. EXIBIÇÃO DO RESULTADO FINAL
            st.success('Sua newsletter foi gerada com sucesso!')
            st.subheader("Visualização da Newsletter")
            st.components.v1.html(newsletter_html, height=600, scrolling=True)
