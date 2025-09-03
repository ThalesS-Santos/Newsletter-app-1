import streamlit as st
import pandas as pd
import requests
import json

# --- BIBLIOTECAS ADICIONADAS ---
try:
    from newsdataapi import NewsDataApiClient
    import google.generativeai as genai
    from pydantic import BaseModel, Field
    from typing import List
    from serpapi import GoogleSearch
    from newsapi import NewsApiClient

except ImportError as e:
    st.error(f"""
        Uma ou mais bibliotecas necessárias não foram encontradas.
        Por favor, instale-as executando o comando abaixo no seu terminal:

        pip install streamlit pandas requests newsdataapi google-generativeai pydantic serpapi-google-search newsapi-python

        Erro original: {e}
    """)
    st.stop()

# --- CHAVES DE API ATUALIZADAS ---
try:
    # Chaves existentes
    NEWS_API_KEY = st.secrets["NEWS_API_KEY"] # Para NewsData.io
    JINA_API_KEY = st.secrets["JINA_API_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    # Novas chaves
    SERPAPI_API_KEY = st.secrets["SERPAPI_API_KEY"]
    NEWSAPI_ORG_KEY = st.secrets["NEWSAPI_ORG_KEY"]

    genai.configure(api_key=GEMINI_API_KEY)
except (KeyError, FileNotFoundError):
    st.error("Erro: Uma ou mais chaves de API não foram encontradas. Verifique seu arquivo .streamlit/secrets.toml.")
    st.stop()

# --- NOVAS FUNÇÕES DE BUSCA ---
COLUNAS_FINAIS = ['title', 'link', 'source']

import pandas as pd
from datetime import datetime, timedelta




def buscar_google_news(termo):
    from GoogleNews import GoogleNews

    # Inicializa o objeto GoogleNews com os parâmetros desejados
    googlenews = GoogleNews(
        lang='pt-BR',        # Define o idioma para português do Brasil
        period='1d',         # Define o período para os últimos 7 dias
        encode='utf-8'       # Define a codificação para UTF-8
    )

    # Realiza a busca por notícias relacionadas ao termo 'tecnologia'
    googlenews.search('Inteligencia Artificial')

    # Define o número máximo de resultados desejados
    max_resultados = 2000
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
    print(f'Quantidade de notícias retornadas: {quantidade_noticias}')

    # Coloca todas as noticias num dataframe
    import pandas as pd
    df = pd.DataFrame(resultados)
    df['link'] = df['link'].str.split('&ved').str[0]
    # a coluna media deve ser renomeada para source
    df.rename(columns={'media': 'source'}, inplace=True)

    return df



def pega_noticias(termo_busca):
    """Busca notícias de múltiplas fontes, combina e remove duplicatas."""

    todas_as_noticias = buscar_google_news(termo_busca)

    if todas_as_noticias.empty:
        return pd.DataFrame()

    # Limpa e remove duplicatas baseadas no link
    todas_as_noticias.dropna(subset=['link'], inplace=True)
    noticias_unicas = todas_as_noticias.drop_duplicates(subset=['link'], keep='first')
    noticias_unicas = noticias_unicas.drop_duplicates(subset=['title'], keep='first')
    #resetaer index
    noticias_unicas.reset_index(drop=True, inplace=True)
    print(noticias_unicas.shape)
    # adicionar um reset_index

    print(f"Busca concluída! {noticias_unicas.shape[0]} notícias únicas encontradas (antes do limite).") # Changed st.success to print
    return  noticias_unicas

# --- FUNÇÕES DE PROCESSAMENTO ---
@st.cache_data(ttl=3600)
def extrair_conteudo_noticias(df_noticias):
    """Extrai o conteúdo completo dos artigos usando a Jina AI API."""
    conteudos = []
    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "X-Engine": "browser"
    }
    
    total_noticias = len(df_noticias)
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Usando enumerate para um contador sequencial (i)
    for i, (index, row) in enumerate(df_noticias.iterrows()):
        status_text.text(f"Extraindo notícia {i + 1}/{total_noticias}: {row['title'][:50]}...")
        url = f"https://r.jina.ai/{row['link']}"
        try:
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            conteudos.append(response.text)
        except requests.exceptions.RequestException as e:
            conteudos.append(f"Erro ao buscar conteúdo para o título '{row['title']}': {e}")
        
        # Usando o contador 'i' para garantir que o valor seja sempre entre 0.0 e 1.0
        progress_bar.progress((i + 1) / total_noticias)
    
    status_text.empty()
    return pd.DataFrame({
        'title': df_noticias['title'],
        'link': df_noticias['link'],
        'content': conteudos
    })

# --- NOVA FUNÇÃO processa_noticias_com_gemini (JÁ INTEGRADA) ---
@st.cache_data(ttl=3600)
def processa_noticias_com_gemini(df_conteudos):
    """Processa o conteúdo das notícias com a API do Gemini para extrair e estruturar dados."""

    # Nota: A classe Noticia não é mais usada na chamada da API, mas serve como documentação.
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

    respostas_json = []
    links_originais = df_conteudos['link'].tolist()

    total_conteudos = len(df_conteudos)
    
    # Os elementos visuais do Streamlit foram removidos desta função.
    # O progresso será impresso no terminal.

    for i, texto in enumerate(df_conteudos['content']):
        print(f"Analisando com IA - Notícia {i + 1}/{total_conteudos}")
        if texto.startswith("Erro ao buscar conteúdo"):
            respostas_json.append(json.dumps({"titulo": "Conteúdo da notícia não disponível"}))
            continue

        try:
            model = genai.GenerativeModel(model_name="gemini-2.5-flash")
            
            response = model.generate_content(
                f"""
                Analise o seguinte texto de uma notícia e extraia as informações no formato JSON.
                O JSON deve seguir a seguinte estrutura:
                {{
                    "titulo": "O título da notícia.",
                    "data_de_publicacao": "A data em que a notícia foi publicada (se disponível).",
                    "resumo_curto": "Um resumo conciso da notícia, apensa com o assunto principal da noticia, não precisa enrolar muito, apenas o basico para um usuario entender do que se trata a noticia, entre 30 palavras e 50 palavras.",
                    "resumo_maior": "Um resumo mais detalhado da notícia, apenas com as informações mais relevantes da noticia e algumas observações a mais, com mais de 150 palavras.",
                    "links_de_imagens": ["Uma lista contendo até 2 URLs das imagens mais relevantes da notícia. Se não houver, retorne uma lista vazia."]
                }}

                Texto da notícia:
                ---
                {texto}
                """,
                generation_config={"response_mime_type": "application/json"}
            )

            noticia_processada = json.loads(response.text)
            noticia_processada['link'] = links_originais[i]
            respostas_json.append(json.dumps(noticia_processada, ensure_ascii=False))

        except Exception as e:
            print(f"Erro ao processar notícia com Gemini: {e}")
            respostas_json.append(json.dumps({"titulo": "Conteúdo da notícia não disponível"}))

    return respostas_json

# --- FUNÇÃO DE RENDERIZAÇÃO ---
def gerar_newsletter_streamlit(lista_json):
    """Renderiza a newsletter na interface do Streamlit."""
    if not lista_json:
        st.info("Nenhuma notícia processada para exibir.")
        return
    
    st.write(f"**Exibindo {len(lista_json)} notícias processadas:**")

    for i, noticia_str in enumerate(lista_json):
        try:
            noticia = json.loads(noticia_str)
            if not noticia or noticia.get("titulo") == "Conteúdo da notícia não disponível":
                continue
        except (json.JSONDecodeError, AttributeError):
            continue

        titulo = noticia.get("titulo", "Título não encontrado")
        data = noticia.get("data_de_publicacao", "Data não informada")
        resumo_curto = noticia.get("resumo_curto", "")
        resumo_maior = noticia.get("resumo_maior", "")
        link = noticia.get("link", "#")
        imagens = noticia.get("links_de_imagens", [])
        imagem = imagens[0] if imagens else "https://via.placeholder.com/400x267?text=Sem+Imagem"

        with st.container():
            col_img, col_content = st.columns([1, 4])
            with col_img:
                st.image(imagem, width='stretch')
            with col_content:
                st.subheader(titulo)
                st.caption(f"Publicado em: {data}")
                st.write(resumo_curto)
                if resumo_maior:
                    with st.expander("Ler mais..."):
                        st.write(resumo_maior)
                st.markdown(f'<a href="{link}" target="_blank">Notícia completa ↗</a>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

# --- INTERFACE PRINCIPAL DO STREAMLIT ---
st.set_page_config(page_title="Gerador de Newsletter com IA", layout="centered")

st.markdown("""
<style>
.st-emotion-cache-1r4qj8v { border: 1px solid #e6e6e6; border-radius: 10px; padding: 1rem 1rem 1rem 1.5rem; margin-bottom: 1rem; box-shadow: 0 2px 6px rgba(0,0,0,0.04); }
.st-emotion-cache-1r4qj8v:hover { transform: translateY(-4px); box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
a { color: #0a9396 !important; font-weight: 500; text-decoration: none; }
a:hover { text-decoration: underline; }
</style>
""", unsafe_allow_html=True)

st.title("📰 Gerador de Newsletter com IA")
st.markdown("Digite um tema, clique em gerar e obtenha um resumo das últimas notícias de múltiplas fontes, processado por Inteligência Artificial.")

termo_busca = st.text_input("Qual tema você quer pesquisar?",)
max_noticias = st.number_input("Número máximo de notícias para a newsletter", min_value=1, max_value=20, value=5, help="Selecione o número de notícias para processar e exibir (máx. 20).")

if st.button("Gerar Newsletter"):
    if not termo_busca:
        st.warning("Por favor, digite um termo para a busca.")
    else:
        df_noticias = pega_noticias(termo_busca, max_noticias)

        if not df_noticias.empty:
            st.info(f"Iniciando processamento de {len(df_noticias)} notícias...")
            
            df_conteudos = extrair_conteudo_noticias(df_noticias)
            
            # Adicionando um spinner para a etapa de processamento com IA
            with st.spinner("A Inteligência Artificial está analisando as notícias... (verifique o terminal para progresso detalhado)"):
                resumos_json = processa_noticias_com_gemini(df_conteudos)
            
            st.success("Newsletter gerada com sucesso!")
            st.markdown("---")
            gerar_newsletter_streamlit(resumos_json)
        else:
            st.error(f"Nenhuma notícia encontrada para o termo '{termo_busca}' em nenhuma das fontes. Tente outro termo.")


