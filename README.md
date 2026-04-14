# tfg-redes-neuronales-PINN-Lombardo

1. El programa requiere exclusivamente el archivo DATA.mat, que contiene las series de temperaturas correspondientes a los distintos dominios y al sensor. Este archivo constituye la entrada de datos necesaria para la correcta ejecución del modelo.
2. Una vez ejecutado, se solicita al usuario 2 cosas:
   2.l. Número de épocas de entrenamiento. El proceso de aprendizaje de la red neuronal converge de forma adecuada a partir de aproximadamente 10.000 épocas.
   2.2 Se incluye la opción de añadir ruido gaussiano blanco a los datos de entrada (valor cuantitativo, si se pone 0 no se aplica ruido), con el objetivo de simular condiciones más realistas y evaluar la robustez del modelo frente a perturbaciones estocásticas.
