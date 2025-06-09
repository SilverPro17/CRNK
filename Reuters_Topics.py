import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

print("📰 Iniciando Exercício 2: Classificação Reuters")
print("="*50)

# Imports necessários
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import reuters
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

print("✅ Bibliotecas importadas com sucesso!")

# 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS
print("\n📥 Carregando dataset Reuters...")
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10000)

print(f"Exemplos de treino: {len(x_train)}")
print(f"Exemplos de teste: {len(x_test)}")
print(f"Número de categorias: {len(set(y_train))}")

# Verificar distribuição das classes
unique, counts = np.unique(y_train, return_counts=True)
print(f"Primeiras 5 categorias mais frequentes:")
for i in range(5):
    print(f"  Categoria {unique[i]}: {counts[i]} exemplos")

# 2. FUNÇÃO DE PRÉ-PROCESSAMENTO (DEFINIDA AQUI!)
def vectorizar_sequences(sequences, dimension=10000):
    """
    Converte listas de integers em matriz binária (one-hot)
    
    Args:
        sequences: Lista de listas de integers
        dimension: Tamanho do vocabulário
    
    Returns:
        Matriz numpy binária
    """
    print(f"🔄 Vetorizando {len(sequences)} sequências...")
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        if sequence:  # Verificar se a sequência não está vazia
            results[i, sequence] = 1.0
    print("✅ Vetorização concluída!")
    return results

# 3. PRÉ-PROCESSAMENTO DOS DADOS
print("\n🔧 Pré-processando dados...")

# Vectorizar dados
x_train = vectorizar_sequences(x_train)
x_test = vectorizar_sequences(x_test)

# Converter labels para one-hot encoding
print("🔄 Convertendo labels para one-hot...")
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(f"✅ Formato dos dados de treino: {x_train.shape}")
print(f"✅ Formato dos labels (one-hot): {y_train.shape}")

# 4. CONSTRUÇÃO DO MODELO
print("\n🏗️ Construindo modelo de rede neural...")
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10000,), name='camada_1'),
    layers.Dense(64, activation='relu', name='camada_2'),
    layers.Dense(46, activation='softmax', name='camada_saida')  # 46 categorias
])

print("✅ Modelo construído!")

# 5. COMPILAÇÃO DO MODELO
print("\n⚙️ Compilando modelo...")
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',  # Para classificação multiclasse
    metrics=['accuracy']
)

# Mostrar arquitetura do modelo
print("\n📋 Arquitetura do modelo:")
model.summary()

# 6. CRIAÇÃO DE CONJUNTO DE VALIDAÇÃO
print("\n📊 Criando conjuntos de validação...")
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = y_train[:1000]
partial_y_train = y_train[1000:]

print(f"✅ Treino: {partial_x_train.shape[0]} exemplos")
print(f"✅ Validação: {x_val.shape[0]} exemplos")
print(f"✅ Teste: {x_test.shape[0]} exemplos")

# 7. TREINAMENTO DO MODELO
print("\n🚀 Iniciando treinamento...")
print("⏰ Isso pode levar alguns minutos...")

history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val),
    verbose=1
)

print("🎉 Treinamento concluído!")

# 8. AVALIAÇÃO DO MODELO
print("\n📈 Avaliando modelo no conjunto de teste...")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"🎯 Acurácia no conjunto de teste: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# 9. FUNÇÃO PARA VISUALIZAR RESULTADOS
def plot_training_history(history):
    """Plota gráficos de loss e accuracy durante o treinamento"""
    print("\n📊 Gerando gráficos de treinamento...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss
    ax1.plot(history.history['loss'], 'b-', label='Treino', linewidth=2)
    ax1.plot(history.history['val_loss'], 'r-', label='Validação', linewidth=2)
    ax1.set_title('Loss do Modelo', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(history.history['accuracy'], 'b-', label='Treino', linewidth=2)
    ax2.plot(history.history['val_accuracy'], 'r-', label='Validação', linewidth=2)
    ax2.set_title('Acurácia do Modelo', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Acurácia')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reuters_training_history.png', dpi=300, bbox_inches='tight')
    print("✅ Gráfico salvo como 'reuters_training_history.png'")
    plt.show()

# Visualizar histórico de treinamento
plot_training_history(history)

# 10. ANÁLISE DETALHADA DAS PREDIÇÕES
print("\n🔍 Analisando predições...")
predictions = model.predict(x_test, verbose=0)

# Converter predições para classes
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

# Calcular accuracy manualmente para verificação
correct_predictions = np.sum(predicted_classes == true_classes)
total_predictions = len(true_classes)
manual_accuracy = correct_predictions / total_predictions

print(f"✅ Predições corretas: {correct_predictions}/{total_predictions}")
print(f"✅ Acurácia calculada manualmente: {manual_accuracy:.4f}")

# 11. MOSTRAR EXEMPLOS DE PREDIÇÕES
print("\n📋 Exemplos de predições:")
print("-" * 60)
for i in range(10):
    is_correct = predicted_classes[i] == true_classes[i]
    confidence = predictions[i][predicted_classes[i]]
    status = "✅ CORRETO" if is_correct else "❌ INCORRETO"
    
    print(f"Exemplo {i+1:2d}: Real={true_classes[i]:2d} | "
          f"Predito={predicted_classes[i]:2d} | "
          f"Confiança={confidence:.3f} | {status}")

# 12. ESTATÍSTICAS GERAIS
print(f"\n📊 ESTATÍSTICAS FINAIS:")
print("=" * 40)
print(f"🎯 Acurácia Final: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"📉 Loss Final: {test_loss:.4f}")
print(f"📈 Melhoria possível: {(1-test_accuracy)*100:.1f} pontos percentuais")

# 13. SALVAR MODELO (OPCIONAL)
try:
    model.save('reuters_classifier_model.h5')
    print(f"💾 Modelo salvo como 'reuters_classifier_model.h5'")
except Exception as e:
    print(f"⚠️ Não foi possível salvar o modelo: {e}")

# 14. RELATÓRIO DE CLASSIFICAÇÃO (TOP 10 CLASSES)
print(f"\n📋 Relatório detalhado (classes mais comuns):")
print("-" * 50)

# Encontrar as 10 classes mais comuns
class_counts = np.bincount(true_classes)
top_10_classes = np.argsort(class_counts)[-10:][::-1]

# Filtrar apenas exemplos das top 10 classes
mask = np.isin(true_classes, top_10_classes) & np.isin(predicted_classes, top_10_classes)

if np.sum(mask) > 10:  # Se temos exemplos suficientes
    try:
        report = classification_report(
            true_classes[mask], 
            predicted_classes[mask],
            target_names=[f'Categoria {i}' for i in top_10_classes],
            digits=3
        )
        print(report)
    except Exception as e:
        print(f"⚠️ Não foi possível gerar relatório detalhado: {e}")

print(f"\n🎉 EXERCÍCIO 2 REUTERS CONCLUÍDO COM SUCESSO!")
print("=" * 50)