import pandas as pd

# для работы с разреженными матрицами
from scipy.sparse import csr_matrix

# метод k-ближайших соседей
from sklearn.neighbors import NearestNeighbors

from pathlib import Path

pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

data_folder = Path("/Users/nikitasurinov/Desktop/Data_bases")
file_to_open_1 = data_folder / "movies.csv"
file_to_open_2 = data_folder / "ratings.csv"
movies = pd.read_csv(file_to_open_1)
ratings = pd.read_csv(file_to_open_2)

# Для прочтения файлов на сторонней машине
# movies = pd.read_csv("movies.csv")
# ratings = pd.read_csv("ratings.csv")


# посмотрим на содержимое файла movies.csv
# дополнительно удалим столбец genres, он нам не нужен
# (параметр axis = 1 говорит, что мы работаем со столбцами, inplace = True, что изменения нужно сохранить)
movies.drop(['genres'], axis=1, inplace=True)
ratings.drop(['timestamp'], axis=1, inplace=True)

# Сделаем матрицу предпочтений с помощью сводной таблицы pivot table. Записью станет каждый фильм, столбцами будут
# пользователи, значениями - рейтинг
user_item_matrix = ratings.pivot(index='movieId', columns='userId', values='rating')

# заполним Nan нулями
user_item_matrix.fillna(0, inplace=True)

# Колонку rating группируем по пользователю (т.е. userId) и считаем результат
# Должна получится таблица в которой каждому пользователю соответствует количество его голосов
users_votes = ratings.groupby(by='userId')['rating'].agg('count')

# Колонку rating группируем по movieId, получится id фильма и сколько человек его оценили
movies_votes = ratings.groupby(by='movieId')['rating'].agg('count')

# Создадим фильтр. Фильтр также называют mask. В результате массив с индексами элементов соответствующих условию.
user_mask = users_votes[users_votes > 50].index  # Индексы юзеров которые голосовали более 50 раз
movie_mask = movies_votes[movies_votes > 10].index  # Индексы фильмов за которые голосовали более 10 раз

# Применим фильтры и отберем фильмы с достаточным количеством оценок.
# Фильтруем строки: то есть movieId накладываем фильтр (то есть список индексов фильмов)
user_item_matrix = user_item_matrix.loc[movie_mask, :]

# Отберем активных пользователей. Столбцы: накладываем фильтр по userId.
user_item_matrix = user_item_matrix.loc[:, user_mask]

# В дата-фрейме много нулей, следовательно, матрица - разреженная (sparse matrix).
# Для того чтобы преодолеть эту сложность, можно преобразовать данные в формат сжатого хранения строкой
# (сompressed sparse row, csr).

# Атрибут values передаст функции csr_matrix только значения дата-фрейма.
csr_data = csr_matrix(user_item_matrix.values)

#  В данном формате сначала записывается положение ненулевого значения, потом само значение.
# print(csr_data[:2,:5])


# Сбросим индексы, так как некоторые строки удалены.
user_item_matrix = user_item_matrix.rename_axis(None, axis=1).reset_index()

# Создадим объект класса NearestNeighbors
# Ближайших соседей - 20, алгоритм bruteforce - грубый перебор, способ вычисления расстояния - косинусное сходство.
# Вычисление ведется на всех свободных ядрах компьютера (-1).
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)

# введем фиксированные параметры: количество рекомендаций и фильм к которому будем их получать
recommendations = 10
search_word = 'Matrix'

# найдем индекс фильма в матрице предпочтений
# найдем фильмы в названиях которых есть искомая подстрока
# каждый заголовок сделать строкой и проверить содержание в нем подстроки search_word
movie_search = movies[movies['title'].str.contains(search_word)]
# print(movie_search)
# вариантов может быть несколько в выдаче, будем брать первый вариант: индекс фильма
movie_id = movie_search.iloc[0]['movieId']

# Далее по индексу фильма в дата-фрейме movies найдем соответствующий индекс в матрице предпочтений
movie_id = user_item_matrix[user_item_matrix['movieId'] == movie_id].index[0]
# В матрице предпочтений не первоначальные индексы, а сброшенные, по этому индекс будет другой.

# Далее с помощью метода knn.kneighbors() найдем индексы ближайших соседей «Матрицы».
distances, indices = knn.kneighbors(csr_data[movie_id], n_neighbors=recommendations + 1)

# В качестве параметров мы передадим:
# csr_data[movie_id], то есть индекс нужного нам фильма из матрицы предпочтений
# n_neighbors, количество соседей; (+1) так как алгоритм также считает расстояние до самого себя

# Массив имеет формат 'numpy.ndarray', нам нужно преобразовать его в список. squeeze() убирает лишние измерения
# лишние измерения в нашем случае это [[1 2 3 4]] N-размерный массив numpy
indices_list = indices.squeeze().tolist()

# это в какой то степени похоже на  сглаживание списка.
# from iteration_utilities import deepflatten
# indices_list = list(deepflatten(indices,1))

distances_list = distances.squeeze().tolist()

# склеим расстояния и индексы и преобразуем в список кортежей
indices_distances = list(zip(indices_list, distances_list))

# сортируем по дистанции по убыванию
indices_distances_sorted = sorted(indices_distances, key=lambda x: x[1], reverse=False)

# Уберем первый элемент, так как это всегда будет сам фильм для поиска соседей
indices_distances_sorted = indices_distances_sorted[1:]

# Теперь по индексам найдем названия фильмов.
# Имеем индексы из матрицы предпочтений, а названия лежат в основном фрейме


recom_list = []

for ind_dist in indices_distances_sorted:
    # искать movieId в матрице предпочтений
    matrix_movie_id = user_item_matrix.iloc[ind_dist[0]]['movieId']
    # получили первоначальный индекс фильма из колонки movieId

    # выяснять индекс этого фильма в датафрейме movies
    id = movies[movies['movieId'] == matrix_movie_id].index

    # брать название фильма и расстояние до него
    title = movies.iloc[id]['title'].values[0]
    dist = ind_dist[1]

    # помещать каждую пару в питоновский словарь
    # который, в свою очередь, станет элементом списка recom_list
    recom_list.append({'Title': title, 'Distance': dist})


# Преобразуем список в фрейм. Индекс будем начинать с 1.
recommendations_df = pd.DataFrame(recom_list, index=range(1, recommendations + 1))
print(recommendations_df)
