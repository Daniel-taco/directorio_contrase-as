import numpy as np

# Función de activación escalón
def step_function(x):
    return 1 if x >= 0 else 0

# Entradas y salidas de la tabla de verdad del AND
inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
outputs = np.array([0, 0, 0, 1])

# Inicialización de los pesos y el sesgo
weights = np.random.rand(2)
bias = np.random.rand(1)

# Tasa de aprendizaje
learning_rate = 0.1

# Función de entrenamiento
def train(inputs, outputs, weights, bias, learning_rate, epochs=100):
    for epoch in range(epochs):
        total_error = 0
        for input_vector, expected_output in zip(inputs, outputs):
            # Calcular la salida del perceptrón
            linear_output = np.dot(input_vector, weights) + bias
            prediction = step_function(linear_output)

            # Calcular el error
            error = expected_output - prediction
            total_error += abs(error)

            # Actualizar los pesos y el sesgo
            weights += learning_rate * error * input_vector
            bias += learning_rate * error

            # Mostrar los detalles de la iteración
            print(f"Iteración: {epoch + 1}, Entrada: {input_vector}, Predicción: {prediction}, "
                  f"Esperado: {expected_output}, Error: {error}, Pesos: {weights}, Sesgo: {bias}")

        # Si no hay errores, terminar el entrenamiento
        if total_error == 0:
            print("El perceptrón ha aprendido la función AND correctamente.")
            break

# Entrenar el perceptrón
train(inputs, outputs, weights, bias, learning_rate)
