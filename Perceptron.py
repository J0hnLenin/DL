import numpy as np
import random
from typing import List, Callable, Union
from numpy.typing import NDArray

class Perceptron:
  def __init__(self, hidden_layers : List[int],
               activations: List[Callable],
               eta:Union[int, float] = 1,
               n_epochs: int = 100,
               batch_size: int = 32,
               random_weights=None,
               random_state=None):
    """
    Инициализирует перцептрон.

    Args:
        hidden_layers (List[int]): Список размеров скрытых слоев.
        activations (List[Callable]): Список функций активации для каждого скрытого слоя.
        eta (Union[int, float]): Скорость обучения.
        n_epochs (int): Количество эпох обучения.
        batch_size (int): Размер мини-пакета.
        random_weights: Начальные веса (если заданы).
        random_state: Зерно для случайных чисел.
    """
    np.random.seed(random_state)
    random.seed(random_state)
    self.is_fitted = False  # Флаг, указывающий, что модель обучена
    self.layers = []         # Список слоев перцептрона
    self.epochs = n_epochs   # Количество эпох обучения
    self.eta = eta           # Скорость обучения
    self.errors = []        # Список ошибок на каждой эпохе
    self.batch_size = batch_size # Размер мини-пакета
    l = hidden_layers + [1]   # Размеры слоев (скрытые + выходной)
    for i in range(len(hidden_layers)):
      self.layers.append(Layer((l[i]+1, l[i+1]), activations[i], i + 1, random_weights=random_weights))


    if not self.layers or self.layers[-1].activation != Perceptron.sigmoid:
        self.layers.append(Layer((l[-1]+1, 1), Perceptron.sigmoid, len(self.layers), random_weights=random_weights))

  @staticmethod
  def relu(x):
      """Функция ReLU (Rectified Linear Unit)"""
      return np.maximum(0, x)
  @staticmethod
  def step(x):
      """Функция step-function (ступенчатая функция)"""
      return x >= 0

  @staticmethod
  def sigmoid(x):
      """Сигмоидальная функция"""
      return 1 / (1 + np.exp(-x))

  @staticmethod
  def shuffle(X, y):

    n = len(y)
    a = [(random.random(), X[i, :], y[i]) for i in range(n)]
    a.sort()
    new_X = np.array([a[i][1] for i in range(n)])
    new_y = np.array([a[i][2] for i in range(n)])
    return new_X, new_y

  @staticmethod
  def get_grad(activation):
    """
    Возвращает функцию для вычисления градиента функции активации.
    """
    if activation == Perceptron.relu or activation == Perceptron.step:
        return lambda x: (x >= 0).astype(int)  # Векторизованное сравнение
    elif activation == Perceptron.sigmoid:
        s = Perceptron.sigmoid
        return lambda x: s(x) * (1 - s(x))
    return Exception("Unknown activation function")

  def predict(self, train_sample: NDArray, logging: bool = False) -> NDArray:
      """
      Предсказывает выход для заданных входных данных.
      """
      result = np.zeros(train_sample.shape[0])
      for i in range(train_sample.shape[0]):
          x = train_sample[i, :]
          for layer in self.layers:
              x = np.append(x, values=[1]) # Добавляем смещение
              x = layer.forward(x, logging)

          result[i] = x[0]
          if logging:
              print(result[i])
      return result

  def train(self, train_sample: NDArray,
            train_ans: NDArray,
            logging=False,
            activation=np.sign,
            random_weights=None) -> list[float]:
      """
      Обучает перцептрон.
      """
      if not self.is_fitted:
          # Инициализация первого слоя (входного)
          first_layer_size = train_sample.shape[1] + 1  # Размер входных данных + смещение
          if len(self.layers) > 1: # Проверяем наличие скрытого слоя
              second_layer_size = self.layers[0].size[0] - 1 # Размер первого скрытого слоя
          else:
              second_layer_size = 1 # Если только выходной слой, то размер 1
          self.layers = [Layer((first_layer_size, second_layer_size),
                                activation,
                                0,
                                random_weights=random_weights)] + self.layers
          self.is_fitted = True  # Отмечаем, что модель обучена

      self.errors = []
      for _ in range(self.epochs):
          train_sample, train_ans = Perceptron.shuffle(train_sample, train_ans) # Перемешиваем данные

          error = 0
          for start in range(0, train_sample.shape[0], self.batch_size):
              # Цикл по мини-пакетам
              end = min(start + self.batch_size, train_sample.shape[0]) # Определяем конец пакета
              batch_X = train_sample[start:end] # Берем входные данные для пакета
              batch_y = train_ans[start:end]   # Берем целевые значения для пакета

              activations = [batch_X]           # Сохраняем входные данные
              x = batch_X
              for layer in self.layers:
                  x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1) # Добавляем смещение ко входу каждого слоя
                  x = layer.forward(x, logging)    # Прямой проход через слой
                  activations.append(x)         # Сохраняем активации слоя

              # Вычисление бинарной кросс-энтропии для пакета
              predicted = x                    # Выход сети (размер пакета, 1)
              loss = -np.mean(batch_y * np.log(predicted) + (1 - batch_y) * np.log(1 - predicted))

              error += loss                     # Суммируем ошибку

              # Обратное распространение ошибки
              delta_next = predicted - batch_y  # Для сигмоидального выхода
              for i in range(len(self.layers) - 1, -1, -1):
                  # Обратный проход по слоям
                  layer = self.layers[i]
                  layer_input = activations[i]
                  delta_next = layer.backprop(delta_next, layer_input, self.eta)
          self.errors.append(error) # Сохраняем ошибку на эпохе
      return self.errors

class Layer:
  def __init__(self,
               size: tuple[int, int],
               activation: Callable,
               index: int,
               value: Union[int, float]=0,
               random_weights=None):
    """
    Инициализирует слой нейронной сети.

    Args:
        size (tuple[int, int]): Размер матрицы весов (входные нейроны, выходные нейроны).
        activation (Callable): Функция активации.
        index (int): Индекс слоя.
        value (Union[int, float], optional): Начальное значение весов. Defaults to 0.
        random_weights: Диапазон случайных весов (если задан).
    """
    self.size = size
    self.w = np.full(size, value) # Инициализируем веса заданным значением
    if random_weights is not None:
      self.w = np.random.randint(random_weights[0], random_weights[1], self.size) # Инициализируем случайными весами
    self.activation = np.vectorize(activation) # Векторизуем функцию активации
    self.i = index # Индекс слоя
    self.last_result = np.array([]) # Последний результат (активации)
    self.last_x = np.array([])    # Последний вход
    self.last_m = np.array([])    # Последняя взвешенная сумма

  def backward(self, value) -> None:
    """
    Устаревший метод обратного распространения (не используется).
    """
    if value > 0:
      # надо увеличить те веса, где нет активации,
      # но должна быть активация
      d = (self.last_x>0)*value
      d = np.repeat(np.array([d]).T, self.w.shape[1], axis=1)
      self.w = self.w + d
    if value < 0:
      # надо уменьшить те веса,
      # где активации быть не должно
      d = (self.last_x>0)*value
      d = np.repeat(np.array([d]).T, self.w.shape[1], axis=1)
      self.w = self.w + d

  def backprop(self, delta_prev, layer_input, eta):
      """
      Выполняет обратное распространение ошибки для слоя.

      Args:
          delta_prev: Ошибка, полученная от следующего слоя.
          layer_input: Входные данные для этого слоя.
          eta: Скорость обучения.

      Returns:
          delta_next: Ошибка для предыдущего слоя.
      """
      grad = Perceptron.get_grad(self.activation)(self.last_x) # (размер пакета, размер слоя)
      delta_w = delta_prev * grad # (размер пакета, размер слоя)

      # Вычисляем градиент весов с использованием матричного умножения
      weight_gradient = layer_input.T @ delta_w  # (размер входа + 1, размер слоя)

      # Обновляем веса
      self.w += eta * weight_gradient / layer_input.shape[0]  # Делим на размер пакета для усреднения градиента

      # Вычисляем ошибку для предыдущего слоя
      delta_next = delta_w @ self.w[:-1].T # (размер пакета, размер предыдущего слоя)

      return delta_next

  def forward(self, x, logging) -> NDArray:
      """
      Выполняет прямой проход через слой.

      Args:
          x: Входные данные.
          logging: Флаг для вывода отладочной информации.

      Returns:
          result: Выходные данные слоя.
      """
      if logging:
          print(f"Слой №{self.i + 1}")
          print(f"Сенсоры: {x}")
          print(f"Размер: {self.size}")

      m = x @ self.w  # (размер пакета, размер слоя)
      result = self.activation(m)  # (размер пакета, размер слоя)
      self.last_result = result
      self.last_x = x
      self.last_m = m

      if logging:
          print(f"Сумматор: {m}")
          print(f"Активация: {result}")
          print(f"Результат размера {result.shape}")

      return result

  Perceptron