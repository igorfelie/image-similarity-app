import os
import torch
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import matplotlib.pyplot as plt

# Função para carregar o modelo CLIP e processar imagens
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

# Função para gerar embeddings das imagens usando CLIP
def generate_image_embeddings(image_paths, model, processor):
    embeddings = []
    for img_path in image_paths:
        img = Image.open(img_path)
        inputs = processor(images=img, return_tensors="pt", padding=True)
        with torch.no_grad():
            features = model.get_image_features(**inputs)
        embeddings.append(features.squeeze(0))  # Remover a dimensão extra de tamanho 1
    return torch.stack(embeddings)

# Função para calcular similaridade de cosseno entre os embeddings
def compute_similarity(query_embedding, image_embeddings):
    query_embedding = query_embedding.unsqueeze(0) if query_embedding.dim() == 1 else query_embedding
    image_embeddings = image_embeddings if image_embeddings.dim() == 2 else image_embeddings.squeeze(1)
    similarity = cosine_similarity(query_embedding.cpu().numpy(), image_embeddings.cpu().numpy())
    return similarity.flatten()

# Função para comparar imagens usando CLIP
def compare_images_clip(query_image_path, image_paths, k=5, threshold=0.90):
    model, processor = load_clip_model()

    # Gerar embeddings para todas as imagens (incluir a imagem de consulta)
    all_image_paths = [query_image_path] + image_paths
    image_embeddings = generate_image_embeddings(all_image_paths, model, processor)

    # Gerar embedding para a imagem de consulta (primeira da lista)
    query_embedding = image_embeddings[0].unsqueeze(0)  # O embedding da consulta será o primeiro

    # Calcular similaridade entre a imagem de consulta e todas as imagens
    similarities = compute_similarity(query_embedding, image_embeddings)

    # Remover a imagem de consulta da lista de imagens para exibição
    similarities = similarities[1:]  # Ignora a consulta em si
    image_paths_to_show = image_paths  # Ignora a imagem de consulta na exibição

    # Filtrar as imagens com similaridade maior que o limiar (0.90)
    similar_indices = [i for i, sim in enumerate(similarities) if sim > threshold]
    
    if not similar_indices:
        st.write(f"Nenhuma imagem com similaridade superior a {threshold} foi encontrada.")
        return [], []

    # Obter as imagens e similaridades filtradas
    filtered_image_paths = [image_paths[i] for i in similar_indices]
    filtered_similarities = [similarities[i] for i in similar_indices]

    return filtered_image_paths, filtered_similarities

# Função para carregar e exibir as imagens mais semelhantes
def show_images_with_names(image_paths, similarities):
    st.write("Imagens mais semelhantes (similaridade > 0.90):")
    
    # Organizar imagens em colunas
    cols = st.columns(len(image_paths))  # cria uma coluna para cada imagem
    
    for i, (img_path, sim) in enumerate(zip(image_paths, similarities)):
        # Exibir a imagem na coluna
        with cols[i]:
            img = Image.open(img_path)
            st.image(img, caption="", use_container_width=True)  # Exibe imagem com tamanho ajustado
            st.write(f"{os.path.basename(img_path)}")  # Exibir nome da imagem
            st.write(f"Similaridade: {sim:.2f}")  # Exibir similaridade

# Streamlit Interface
def main():
    st.title("Ferramenta de Análise de Similaridade de Imagens")

    # Upload de imagens
    uploaded_files = st.file_uploader("Carregue as imagens do álbum", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

    if uploaded_files:
        # Salvar as imagens carregadas em um diretório temporário
        temp_folder = "temp_images"
        os.makedirs(temp_folder, exist_ok=True)

        image_paths = []
        for uploaded_file in uploaded_files:
            img_path = os.path.join(temp_folder, uploaded_file.name)
            with open(img_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            image_paths.append(img_path)

        # Seleção da imagem de consulta
        query_image_path = st.selectbox("Escolha a imagem de consulta", image_paths)

        # Exibir miniatura da imagem de consulta
        if query_image_path:
            query_image = Image.open(query_image_path)
            st.image(query_image, caption="Imagem de consulta", width=200)  # Exibe miniatura da imagem de consulta

            # Realizar a busca por imagens semelhantes usando CLIP
            similar_image_paths, similarities = compare_images_clip(query_image_path, image_paths)

            # Exibir as imagens mais semelhantes com seus nomes
            if similar_image_paths:
                show_images_with_names(similar_image_paths, similarities)
            else:
                st.write("Nenhuma imagem com similaridade superior a 0.90 foi encontrada.")
    else:
        st.warning("Por favor, carregue algumas imagens para análise.")

if __name__ == "__main__":
    main()
