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

    total_conteudos = len(df_conteudos)
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, texto in enumerate(df_conteudos['content']):
        status_text.text(f"Analisando com IA - Notícia {i + 1}/{total_conteudos}")
        if texto.startswith("Erro ao buscar conteúdo"):
            respostas_json.append(json.dumps({"titulo": "Conteúdo da notícia não disponível"}))
            progress_bar.progress((i + 1) / total_conteudos)
            continue
        
        try:
            # --- MODIFICAÇÃO PRINCIPAL AQUI ---
            # Limita o texto de entrada para evitar exceder os limites da API
            texto_limitado = texto[:15000]

            model = genai.GenerativeModel(model_name="gemini-1.5-flash")
            response = model.generate_content(
                f"Analise o seguinte texto de uma notícia e extraia as informações no formato JSON, conforme o schema solicitado. Texto da notícia:\n\n---\n\n{texto_limitado}",
                # Adiciona um limite de tokens de saída como uma boa prática
                generation_config={
                    "response_mime_type": "application/json",
                    "max_output_tokens": 4096 
                },
                tools=[Noticia]
            )
            
            noticia_processada = json.loads(response.text)
            noticia_processada['link'] = links_originais[i]
            respostas_json.append(json.dumps(noticia_processada, ensure_ascii=False))

        except Exception as e:
            # Mensagem de aviso mais informativa
            st.warning(f"Erro ao processar notícia com Gemini (pode ser devido ao tamanho ou conteúdo do artigo). Link: {links_originais[i]}. Erro: {e}")
            respostas_json.append(json.dumps({"titulo": "Conteúdo da notícia não disponível"}))
        
        progress_bar.progress((i + 1) / total_conteudos)
    
    status_text.empty()
    return respostas_json
