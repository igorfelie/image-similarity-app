import cv2
import numpy as np
from PIL import Image

# Função para extrair descritores SIFT de uma imagem
def extract_sift_features(image_paths):
    sift = cv2.SIFT_create()  # Criar o detector SIFT
    descriptors_list = []
    
    for path in image_paths:
        print(f"Processing image: {path}")
        
        try:
            # Ler a imagem
            img = cv2.imread(path)
            if img is None:
                print(f"Error reading image {path}")
                continue  # Se não conseguir ler, pula a imagem
            
            # Converter para escala de cinza
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detectar keypoints e descritores SIFT
            keypoints, descriptors = sift.detectAndCompute(gray_img, None)
            if descriptors is not None:
                descriptors_list.append(descriptors)
            else:
                print(f"No descriptors found in {path}")
                
        except Exception as e:
            print(f"Error processing image {path}: {e}")
            continue
    
    return descriptors_list
