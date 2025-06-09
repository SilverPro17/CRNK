import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS
from tensorflow.keras.datasets import imdb

# Carregar dataset (top 10000 palavras mais frequentes)
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

print(f"Exemplos de treino: {len(x_train)}")
print(f"Exemplos de teste: {len(x_test)}")
print(f"Exemplo de review: {x_train[0][:10]}...")  # Primeiras 10 palavras

# 2. PRÉ-PROCESSAMENTO
def vectorizar_sequences(sequences, dimension=10000):
    """Converte listas de integers em matriz binária"""
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0
    return results

# Converter para formato one-hot
x_train = vectorizar_sequences(x_train)
x_test = vectorizar_sequences(x_test)

# Converter labels para float32
y_train = np.asarray(y_train).astype('float32')
y_test = np.asarray(y_test).astype('float32')

print(f"Formato dos dados de treino: {x_train.shape}")
print(f"Formato dos labels: {y_train.shape}")

# 3. CONSTRUÇÃO DO MODELO
model = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=(10000,)),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Sigmoid para classificação binária
])

# 4. COMPILAÇÃO DO MODELO
model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',  # Para classificação binária
    metrics=['accuracy']
)

# Visualizar arquitetura
model.summary()

# 5. CRIAÇÃO DE CONJUNTO DE VALIDAÇÃO
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# 6. TREINAMENTO DO MODELO
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val),
    verbose=1
)

# 7. AVALIAÇÃO DO MODELO
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\nAcurácia no conjunto de teste: {test_accuracy:.4f}")

# 8. VISUALIZAÇÃO DOS RESULTADOS
def plot_training_history(history):
    """Plota gráficos de loss e accuracy durante o treinamento"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(history.history['loss'], label='Treino')
    ax1.plot(history.history['val_loss'], label='Validação')
    ax1.set_title('Loss do Modelo')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Accuracy
    ax2.plot(history.history['accuracy'], label='Treino')
    ax2.plot(history.history['val_accuracy'], label='Validação')
    ax2.set_title('Acurácia do Modelo')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Acurácia')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

plot_training_history(history)

# 9. FAZENDO PREDIÇÕES
def predict_review(model, review_text_vector):
    """Faz predição para um review específico"""
    prediction = model.predict(review_text_vector.reshape(1, -1))
    sentiment = "Positivo" if prediction[0] > 0.5 else "Negativo"
    confidence = prediction[0][0] if prediction[0] > 0.5 else 1 - prediction[0][0]
    return sentiment, confidence

# Exemplo de predição
sample_prediction = predict_review(model, x_test[0])
print(f"Predição para primeiro review de teste: {sample_prediction[0]} (confiança: {sample_prediction[1]:.2f})")
print(f"Label real: {'Positivo' if y_test[0] == 1 else 'Negativo'}")