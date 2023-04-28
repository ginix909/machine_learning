import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

df = pd.read_csv('2023_rankings.csv')

# сколько колонок во фрейме
# print(len(list(df.columns.values)))
# print(f'{df.shape[1]} - колонок')
# print(f'{df.shape[0]} - строк')

# выведем численные колонки (численные признаки)
# print(df.select_dtypes(include='integer').columns.values.tolist())
# print(len(df.select_dtypes(include='integer').columns.values.tolist()))

# выведем нечисленные колонки (категориальные признаки)
# print(df.select_dtypes(include='object').columns.values.tolist())
# print(len(df.select_dtypes(include='object').columns.values.tolist()))

# выведем список всех колонок
# print(df.columns.values.tolist())

# сравним значения двух колонок с выводом результата в третью
# df['equal'] = df['rank_order'].equals(df['scores_overall_rank'])
# проверим есть значения False в результирующей колонке equal
# print(df[['equal']].value_counts())
# Только значения  True - ЗНАЧИТ КОЛОНКИ ИДЕНТИЧНЫ, ОДНУ УДАЛЯЕМ!!!
df.drop(['scores_overall_rank'], axis=1, inplace=True)

# проверим количество пропущенных значений во всем фрейме
# print(df.info())

# проверим средние показатели численных данных. Проверка показывает что min значение некоторых признаков 0, что также
# подтверждает нулевые значения.
# print(df.describe())

# Посмотрели распределение университетов по странам
location_counts = df['location'].value_counts()    #посчитали сколько универов в каждой стране
# # нарисовали график пирог со значениями кол универов в стране и подписями их названиями
plt.pie(location_counts, labels=location_counts.index, autopct='%1.1f%%')
plt.axis('equal')                                 # уровнять масштаб оси Х и У
plt.show()

# Изучить распределение по признаку industry_income
high_income = df.sort_values(by=['scores_industry_income_rank'], ascending=False)[:1000]
location_counts = high_income['location'].value_counts()    #посчитали сколько универов в каждой стране
# # нарисовали график пирог со значениями кол универов в стране и подписями их названиями
plt.pie(location_counts, labels=location_counts.index, autopct='%1.1f%%')
# ВЫВОД: срез первых 50 университетов по рейтингу industry_income показывает что Россия занимает первое место - 12%
# от первых 50 университетов по данному показателю
plt.show()

# посчитать есть ли закрытые университеты и есть ли неакредитованные
# print(df[['closed','unaccredited']].value_counts())
df = df.loc[df['unaccredited'] == False]
df.drop(['closed','unaccredited'], axis=1, inplace=True)
# осталось 2344 строки и 21 признак
# выберу одну целевую переменную - Rank - постараюсь предсказать место по Стране и статистическим признакам, предлагаемым
# предметам

# удалим колонки с рейтингом по основным направлениям Обучение, Исследования, Цитируемость, Доход,
# Несмотря на то, что данные колонки явно ощутимо коррелируют с целевой переменной, но они в данном случае
# напрямую влияют на Рейтинговую позицию, так как она из них и счиатается, а я хочу исследовать влияние других факторов
df.drop(['scores_teaching','scores_teaching_rank','scores_research','scores_research_rank','scores_citations',
         'scores_citations_rank','scores_industry_income','scores_industry_income_rank','scores_international_outlook',
         'scores_international_outlook_rank'], axis=1, inplace=True)

# удалим строки после 1508 позиции, так как у них нет правильно посчитанной позиции, то есть их позиция в рейтинге
# фиктивна. Так как после 1508 строки не считается общий балл, точнее он представлен в общем для всех почти 1000 записей
# промежутке.
df = df.iloc[:1508,:]
df.drop(['rank'], axis=1, inplace=True)
# Целевой переменной будет моя колонка my_rank, она будет представлена в обычном порядковом виде без повторений
new_col = df['rank_order']//10

df.insert(loc = 1,
          column = 'my_rank',
          value = new_col)
df.drop(['rank_order'], axis=1, inplace=True)
# print(df[['subjects_offered']].head(10))

# убрали пропущенные значения naN
df = df.dropna()
missing_data_columns = df.isnull().sum()
# print(missing_data_columns)
# Вопрос ! нужно ли скинуть индексы после этого ?
df = df.reset_index(drop=True)
# Я сбросил индексы. пока не вижу проблемы. Если что, заменим Nan  в колонке subjects_offered на пустой список и сохраним
# количество строк соосно рейтингу, неясно опять же зачем такое может понадобится

# распарсим данные в колонке subjects_offered и представим их списком
# df['my_subjects'] = df[['subjects_offered']]
df['subjects_quantity']= df['subjects_offered'].str.split(',') #распарсили строку в список предметов
df['subjects_quantity'] = df['subjects_quantity'].apply(lambda x: len(x))

# сделать из столбца male_ratio два столбика female_ratio, male_ratio
female_ratio = df['stats_female_male_ratio'].apply(lambda x: x[:2])
male_ratio = df['stats_female_male_ratio'].apply(lambda x: x[5:7])

df.insert(loc = 8,
          column = 'female_ratio',
          value = female_ratio)
df.insert(loc = 9,
          column = 'male_ratio',
          value = male_ratio)
df.drop(['stats_female_male_ratio','subjects_offered'], axis=1, inplace=True)

# Если учитывать что у каждой записи есть индекс, то по сути имена как обычно ничего не дают и их надо удалить
df.drop(['name','aliases'], axis=1, inplace=True)

# print(df['aliases'].value_counts())
# колонка aliasses псевдонимы также как имя не имеет смысла


print(df.head())
# print(df.tail(1200))

# В дальнейшем необходимо создать модель машинного обучения и попытаться предсказывать место университета в рейтинге по
# текущим свободным признакам