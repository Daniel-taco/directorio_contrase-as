import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.zeros(input_size + 1)
        self.learning_rate = learning_rate

    def activate(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return self.activate(summation)

    def train(self, training_inputs, labels):
        epoch = 0
        while True:
            all_correct = True
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                if prediction != label:
                    self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                    self.weights[0] += self.learning_rate * (label - prediction)
                    all_correct = False
            predictions = [self.predict(inputs) for inputs in training_inputs]
            epoch += 1
            print(f"Iteración {epoch}: Predicciones = {predictions}")
            if all_correct:
                break
        return epoch

# Tabla de verdad del operador AND
training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([0, 0, 0, 1])

perceptron = Perceptron(input_size=2)
epochs = perceptron.train(training_inputs, labels)

# Imprimimos la predicción final y el número de iteraciones necesarias
print(f"\nEl perceptrón se entrenó correctamente en {epochs} iteraciones.")
print("Predicción Final:")
for inputs in training_inputs:
    print(f"Entrada: {inputs}, Predicción: {perceptron.predict(inputs)}")
