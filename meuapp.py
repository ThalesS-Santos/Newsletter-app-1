import streamlit as st
import pandas as pd
import requests
import json

try:
    from newsdataapi import NewsDataApiClient
    import google.generativeai as genai
    from pydantic import BaseModel, Field
    from typing import List
except ImportError as e:
    st.error(f"""
        Uma ou mais bibliotecas necessárias não foram encontradas.
        Por favor, instale-as executando o comando abaixo no seu terminal:
        
        pip install streamlit pandas requests newsdataapi google-generativeai pydantic

        Erro original: {e}
    """)
    st.stop()
 

try:
    NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
    JINA_API_KEY = st.secrets["JINA_API_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

    genai.configure(api_key=GEMINI_API_KEY)
except (KeyError, FileNotFoundError):
    st.error("Erro: As chaves de API não foram encontradas. Verifique seu arquivo .streamlit/secrets.toml.")
    st.stop()



@st.cache_data(ttl=3600) # Cache por 1 hora
def pega_noticias(termo_busca):
    """Busca notícias usando a NewsDataApiClient e retorna um DataFrame."""
    try:
        api = NewsDataApiClient(apikey=NEWS_API_KEY)
        response = api.latest_api(q=termo_busca, language='pt', country='br')
        resultados = response.get('results', [])
        
        if not resultados:
            return pd.DataFrame()
            
        df = pd.DataFrame(resultados)
        
        colunas_mapeadas = {'title': 'title', 'link': 'link', 'pubDate': 'published'}
        colunas_existentes = [col for col in colunas_mapeadas.keys() if col in df.columns]
        df_final = df[colunas_existentes].rename(columns=colunas_mapeadas)
        
        return df_final.dropna(subset=['link']).head(5)

    except Exception as e:
        st.error(f"Ocorreu um erro ao buscar notícias: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def extrair_conteudo_noticias(df_noticias):
    """Extrai o conteúdo completo dos artigos usando a Jina AI API."""
    conteudos = []
    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "X-Engine": "browser"
    }
    
    for index, row in df_noticias.iterrows():
        url = f"https://r.jina.ai/{row['link']}"
        try:
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            conteudos.append(response.text)
        except requests.exceptions.RequestException as e:
            conteudos.append(f"Erro ao buscar conteúdo para o título '{row['title']}': {e}")
            
    return pd.DataFrame({
        'title': df_noticias['title'],
        'link': df_noticias['link'],
        'content': conteudos
    })

@st.cache_data(ttl=3600)
def processa_noticias_com_gemini(df_conteudos):
    """Processa o conteúdo das notícias com a API do Gemini para extrair e estruturar dados."""
    
    class Noticia(BaseModel):
        titulo: str = Field(..., description="O título da notícia.")
        data_de_publicacao: str = Field(..., description="A data em que a notícia foi publicada (se disponível).")
        resumo_curto: str = Field(..., description="Um resumo conciso da notícia em até 30 palavras.")
        resumo_maior: str = Field(..., description="Um resumo mais detalhado da notícia em até 150 palavras.")
        links_de_imagens: List[str] = Field(..., description="Uma lista contendo até 2 URLs das imagens mais relevantes da notícia. Se não houver, retorne uma lista vazia.")

    respostas_json = []
    links_originais = df_conteudos['link'].tolist()

    for i, texto in enumerate(df_conteudos['content']):
        if texto.startswith("Erro ao buscar conteúdo"):
            respostas_json.append(json.dumps({"titulo": "Conteúdo da notícia não disponível"}))
            continue
        
        try:
            model = genai.GenerativeModel(model_name="gemini-1.5-flash")
            response = model.generate_content(
                f"Analise o seguinte texto de uma notícia e extraia as informações no formato JSON, conforme o schema solicitado. Texto da notícia:\n\n---\n\n{texto}",
                generation_config={"response_mime_type": "application/json"},
                # A API do Gemini usa `tools` para definir o schema de saída JSON
                tools=[Noticia]
            )
            
            noticia_processada = json.loads(response.text)
            noticia_processada['link'] = links_originais[i]
            respostas_json.append(json.dumps(noticia_processada, ensure_ascii=False))

        except Exception as e:
            st.warning(f"Erro ao processar notícia com Gemini: {e}")
            respostas_json.append(json.dumps({"titulo": "Conteúdo da notícia não disponível"}))
            
    return respostas_json


def gerar_newsletter_streamlit(lista_json):
    """Renderiza a newsletter na interface do Streamlit."""
    if not lista_json:
        st.info("Nenhuma notícia processada para exibir.")
        return

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
                st.image(imagem, use_container_width='always')
            with col_content:
                st.subheader(titulo)
                st.caption(f"Publicado em: {data}")
                st.write(resumo_curto)
                if resumo_maior:
                    with st.expander("Ler mais..."):
                        st.write(resumo_maior)
                st.markdown(f'<a href="{link}" target="_blank">Notícia completa ↗</a>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)



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
st.markdown("Digite um tema, clique em gerar e obtenha um resumo das últimas notícias, processado por Inteligência Artificial.")

termo_busca = st.text_input("Qual tema você quer pesquisar?",)

if st.button("Gerar Newsletter"):
    if not termo_busca:
        st.warning("Por favor, digite um termo para a busca.")
    else:
        with st.spinner("Buscando as notícias mais recentes... ⏳"):
            df_noticias = pega_noticias(termo_busca)

        if not df_noticias.empty:
            with st.spinner("Extraindo o conteúdo completo dos artigos... 📄"):
                df_conteudos = extrair_conteudo_noticias(df_noticias)
            
            with st.spinner("A mágica da IA está acontecendo... Gerando resumos... ✨"):
                resumos_json = processa_noticias_com_gemini(df_conteudos)
            
            st.success("Newsletter gerada com sucesso!")
            st.markdown("---")
            gerar_newsletter_streamlit(resumos_json)
        else:
            st.error(f"Nenhuma notícia encontrada para o termo '{termo_busca}'. Tente outro.")