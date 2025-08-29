# Facial Expression Recognition Using Deep Learning Neural Network

Este repositório reúne a estrutura da documentação acadêmica do projeto, contemplando os tópicos a serem desenvolvidos, bem como todo o referencial teórico utilizado para fundamentar sua elaboração.

O projeto propõe o desenvolvimento de um modelo de Rede Neural Artificial (RNA) para a classificação de emoções a partir de expressões faciais de pacientes com limitações na capacidade de comunicação verbal. O objetivo é oferecer suporte à equipe médica durante o acompanhamento clínico, permitindo a identificação automática de estados emocionais como felicidade, tristeza, raiva, neutralidade e medo. Espera-se que o sistema contribua para uma avaliação mais precisa do bem-estar do paciente, auxiliando na personalização do tratamento, no monitoramento contínuo e na tomada de decisões médicas mais assertivas.

---

🔧 Frameworks e Bibliotecas Principais
- PyTorch → Framework principal para criação, treino e inferência do modelo.
- Torchvision → Transforms (pré-processamento de imagens) e modelos pré-treinados (ResNet, MobileNetV2, Inception, etc.).
- NumPy → Manipulação de arrays e tensores.
- Matplotlib / Seaborn → Visualização de métricas, gráficos de treino e matriz de confusão.
- Scikit-learn → Avaliação do modelo (classification report, confusion matrix, métricas).
- OpenCV (cv2) → Detecção e pré-processamento de rostos (HaarCascade, MTCNN ou MediaPipe).
- Pandas → Organização e análise de dados (opcional para relatórios de resultados).

🌐 Ambiente de Desenvolvimento
- Python 3.x
- Jupyter Notebook ou VS Code para experimentação.
- Git/GitHub → Controle de versão e documentação.

---

## 📌 Sequência de Escrita da Documentação Acadêmica

--- 

## Introdução
### Objetivos Gerais
### Objetivos Específicos
### Justificativa

---

## Referencial Teorico

### Modelo de RNA
- Emoção
- Visão Computacional
- Visão Computacional na Medicina
- Pre-Processamento de Imagens
- Redes Neurais Artificiais
- Redes Neurais de Convolução
- Aprendizado Profundo
- Metricas de Avaliação para redes de classificação

### Deployment do Modelo de RNA
- API (Application Programming Interface)
- Frontend
- Backend
- Python 3.10
- FastAPI
- JWT (JavaScript Web Token)
- Sistema Operacional Linux
- Nginx
- Proxy Reverso
- Docker e Docker Compose
- Banco de Dados Relacional
- PostgreSQL

---

## Desenvolvimento
### Dataset
- FER+ Dataset
- Tipo do Dataset
- Tamanho do Dataset
- Numero de Classes
- Estrutura do Dataset

### Preprocessamento e Carregar Dataset
- Definir o Data Transfom: transformer.Compose(Resize, Flip, Crop, ToTensor, Normalize)
- Carregar o Dataset do diretório: ImageFolder(path_to_dir)
- Data Load: Dividir o Dataset para treino, validação e teste (train_loader, val_loader, test_loader)

### Verificar e Visualizar as o Dataset
- Verificar as variaveis do Dataset: shape do Tensor, classes, numero de classes, size dos dados
- Visualização do Dataset: Use matplotlib para visualização das imagens

### Modelo 
- Criar o Modelo de RNA: class Net(nn.Module) e mover para a GPU
- Transfer Learning: utilizar modelos ja treinados como, inceptionV3, resnet, mobilenetV2, etc.

### Treino
- Loss Function: nn.CrossEntropyLoss() para classificação.
- Optimizer: AdamW, Adam ou SGD.
- Scheduler: StepLR, ReduceLROnPlateau (boa prática).
- Hiperparâmetros: Batch_size, Epochs, LR definidos claramente.
- Salvar métricas: importante para visualização e comparação.
- Salvar modelo: torch.save(model.state_dict(), "modelo.pth").
  
### Avaliação das Metricas
- Grafico de Acuracia de treino e Validação por Epocas
- Grafico de Loss de treino e Validação por Epocas

### Avaliar o Modelo com as Images de Test
- with torch.no_grad(): Desabilitar os cálculos para os gradientes
- Salvar os previsões em labels_true e labels_pred
- Criar a matriz de confusão: sklearn confusion matrix
- Visualizar com seaborn
- Classification Report: precision, recall, f1-score, accuracy

### Inferência
- Carregar modelo treinado: model.load_state_dict(torch.load(...)).
- Detecção de faces: Haarcascade com OpenCV é válido (pode também usar MTCNN ou mediapipe para maior precisão).
- Funções de visualização: importante para validação.
- Transformação da imagem: mesma do treinamento (transform).
- Tensor shape (1, 3, 224, 224): correto.
- Mapeamento de classes: dicionário {0:"Raiva",1:"Feliz",...}.
- Inferência: torch.argmax(output, dim=1).item() ✅.

---

## Contribuições

---

## Conclusão / Resultados

---

## Trabalhos Futuros

---

## Bibliografia

---

## 📜 License

This project is licensed under the **MIT License**.

---
