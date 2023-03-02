import pandas as pd
import numpy as np
from iteration_utilities import deepflatten
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option ('display.max_rows', 3000)
pd.set_option ('display.max_columns', 3000)

def transform_features(df):
    '''Трансформация признаков. функция удалит лишние колонки по смыслу, колонки с большим % пропусков,
     небольшой % пропусков заменит значением моды.'''
    df_copy = df.copy() #создаем копию направленного в функцию массива
    cutoff = df_copy.shape[0] * 0.05 #вычисляем точку останова. 5% от числа строк (от общего колич значений выборки).

    #drop numerical columns with missing data
    numerical_cols = df_copy.select_dtypes(include = ['integer','float']) #выбираем численные колонки
    missing_numerical = numerical_cols.isnull().sum() #подсчет сколько нулевых значений в каждой колонке
    drop_numerical_columns = [] # колонки которые удалим по условию
    for x in missing_numerical.index.values.tolist():
        if missing_numerical[x] > cutoff:# если колич пропусков больше чем cutoff,добавить в список для удаления
            drop_numerical_columns.append(x)
    df_copy.drop(columns = drop_numerical_columns, inplace = True) #удалили колонки с большим кол пропусков

    #удаляем категориальные переменные с пропущенными значениями
    categorical_data = df_copy.select_dtypes(include = ['object'])
    categorical_missing_data = categorical_data.isnull().sum().sort_values(ascending = False)  #подсчет пропущ знач
    drop_columns = categorical_missing_data[categorical_missing_data > 0] #удаляем колонки в которых есть хоть 1 пропуск
    cat_cols = drop_columns.index.values.tolist()
    df_copy.drop(columns = cat_cols , inplace=True)

    #заменим оставшиеся пропущенные значения с помощью сводной статистики
    #пропущенные значения остались только в количесвтенных колонках
    numerical = df_copy.select_dtypes(include = ['integer','float'])
    numerical_cols_impute = numerical.loc[:,(numerical.isnull().sum() > 0)].columns #имена колонок, в которых пропуски
    numerical_mode = df_copy[numerical_cols_impute].mode().to_dict(orient = 'records')[0] # ищем моду каждой колонки
    # делаем из фрейма словарь (список,состоящий из 1 словаря, по этому [0]).
    df_copy.fillna(numerical_mode, inplace = True) #заменяем пропуски 0.

    #feature engineering - добавление новых колонок.
    # год реставрации и год постройки переводим в "лет с последней реставрации"
    df_copy['year_until_remod'] = df_copy['Year Remod/Add'] - df_copy['Year Built']
    df_copy.drop(index = 850, inplace = True) # в этой строке -1 год с момента последней реставрации. удалим.
    df_copy.drop(columns = ['Year Remod/Add','Year Built'], inplace = True)
    df_copy.drop(columns = ['Order','PID','Mo Sold','Yr Sold','Sale Type','Sale Condition'], inplace = True)
    #удаляем колонки по сделке. решили что они не влияют на цену продажи.

    return df_copy

def select_features(df_copy, correlation_cutoff, uniqueness_cutoff):
    '''Отбор признаков. Функция исходя из корреляции каждого признака с исследуемым, отберет те, которые ощутимо
    коррелируют. Если два сильно коррелируют между собой - оставит один.
    correlation_cutoff- отсечка величины корреляции. чтобы удалить колонки меньшей корреляции с целевым признаком.
    uniqueness_cutoff - отсечка для количества категорий значений в колонке. чтобы удалить перенагруженные колонки.'''
    #two factors to consider - Correlation with target column and correlation with other features
    df_copy2 = df_copy.copy()
    df_copy2 = df_copy2.drop(columns = ['Garage Cars','TotRms AbvGrd'])

    #select numerical cols and base on correlation values decide which columns to keep
    numerical_cols = df_copy2.select_dtypes(include = ['integer','float'])
    numerical_correlation = numerical_cols.corr()
    target_correl = numerical_correlation['SalePrice'].abs().sort_values()
    corr_below_cutoff = target_correl[target_correl < correlation_cutoff].index.values.tolist()
    df_copy2 = df_copy2.drop(columns = corr_below_cutoff)

    #select categorical columns and convert them to numerical variables
    #dropping columns with alot of unique values
    categorical_only = df_copy2.select_dtypes(include = ['object'])
    unique_heavy_columns = [col for col in categorical_only.columns if len(categorical_only[col].value_counts()) > uniqueness_cutoff]
    df_copy2 = df_copy2.drop(columns = unique_heavy_columns)

    #converting the remaining categorical columns to dummies
    categorical_columns_only = df_copy2.select_dtypes(include = ['object'])
    for columns in categorical_columns_only.columns:
        df_copy2[columns] = df_copy2[columns].astype('category')

    #converting to dummies
    for columns in categorical_columns_only.columns:
        dummies = pd.get_dummies(df_copy2[columns])
        df_copy2 = pd.concat([df_copy2,dummies], axis = 1)
        del df_copy2[columns]

    return df_copy2

def train_and_test(df_copy2,k):
    '''Тренировка и тестирование. Функция разделит фрейм на тренировочное и тестовое множество
    k - пропорция для разделения фрейма на тренир и тестовое множество'''
    df_copy3 = df_copy2.copy()
    numeric = df_copy3.select_dtypes(include = ['integer','float'])
    columns_numeric = numeric.drop(columns = ['SalePrice'])#только количественные переменные без целевой.
    cols = columns_numeric.columns.values.tolist() #список названий только количественных колонок
    lr = LinearRegression() #классификатор

    if k == 0: #поделим фрейм пополам вручную
        train = df_copy3[0:1460]
        test = df_copy3[1460:]
        lr.fit(train[cols], train['SalePrice'])
        predictions = lr.predict(test[cols])
        mse = mean_squared_error(test['SalePrice'],predictions)
        mae = mean_absolute_error(test['SalePrice'],predictions)
        mape = mean_absolute_percentage_error(test['SalePrice'],predictions)
        R_2 = r2_score(test['SalePrice'],predictions)
        rmse = mse ** (1/2) #корень среднеквадратичной ошибки. на сколько предсказания отличаются от фактических знач.

        return rmse,mse,mae,mape,R_2,predictions,test['SalePrice']

    if k == 1:
        shuffle_df = df_copy3.sample(frac = 1,random_state=42)
        shuffle_df = shuffle_df.reset_index()
        shuffle_df = shuffle_df.drop(columns = ['index'])

        #When we call the reset_index function, a new index column is added. The rows are still sorted.
        #Hence we decided to drop the index column

        fold_one = shuffle_df[0:1460]
        fold_two = shuffle_df[1460:]

        #учим на fold_one, предсказываем для fold_two
        lr.fit(fold_one[cols],fold_one['SalePrice'])
        predictions_one = lr.predict(fold_two[cols])
        mse_one = mean_squared_error(fold_two['SalePrice'], predictions_one)
        mae_one = mean_absolute_error(fold_two['SalePrice'], predictions_one)
        mape_one = mean_absolute_percentage_error(fold_two['SalePrice'], predictions_one)
        R_2_one = r2_score(fold_two['SalePrice'], predictions_one)
        rmse_one = mse_one ** (1/2)

        #учим на fold_two, предсказываем для fold_one
        lr.fit(fold_two[cols], fold_two['SalePrice'])
        predictions_two = lr.predict(fold_one[cols])
        mse_two = mean_squared_error(fold_one['SalePrice'], predictions_two)
        mae_two = mean_absolute_error(fold_one['SalePrice'], predictions_two)
        mape_two = mean_absolute_percentage_error(fold_one['SalePrice'], predictions_two)
        R_2_two = r2_score(fold_one['SalePrice'], predictions_two)
        rmse_two = mse_two ** (1/2)

        avg_rmse = np.mean([rmse_one,rmse_two])
        mse = np.mean([mse_one,mse_two])
        mae = np.mean([mae_one,mae_two])
        mape = np.mean([mape_one,mape_two])
        R_2 = np.mean([R_2_one,R_2_two])

        return avg_rmse, mse,mae,mape,R_2

    else:
        #if k is more than one, then we perform KFold cross validation
        kf = KFold(n_splits = k, random_state=42, shuffle=True) #количество разбиений совокупности
        #можно было просто k передать в cv. так то ли лучше, то ли потому что еще перемешивает.
        mse = cross_val_score(lr,df_copy3[cols], df_copy3['SalePrice'], scoring = 'neg_mean_squared_error', cv = kf)
        mae = cross_val_score(lr,df_copy3[cols], df_copy3['SalePrice'], scoring = 'neg_mean_absolute_error', cv = kf)
        mape = cross_val_score(lr,df_copy3[cols], df_copy3['SalePrice'], scoring = 'neg_mean_absolute_percentage_error', cv = kf)
        R_2 = cross_val_score(lr,df_copy3[cols], df_copy3['SalePrice'], scoring = 'r2', cv = kf)
        rmses = np.sqrt(np.absolute(mse))# все значения идут чистыми, а тут вычисляется корень.
        #оправдать это можно только тем что rmse это отдельная метрика
        avg_rmse_k_fold = np.mean(rmses) #среднее знач точности по всем n выборкам кросс-валидации

        return avg_rmse_k_fold,mse,mae,mape,R_2


final_test_data = pd.read_table('AmesHousing.tsv', delimiter='\t')
transformed_data = transform_features(final_test_data)
selected_features_data = select_features(transformed_data, 0.3, 10)

k=6

def metrics_lists(k):
    '''создаст массивы метрик по каждому разбиению k в кросс-валидации '''
    rmse_list = []
    mse_list = []
    mae_list = []
    mape_list = []
    R_2_list = []
    k_list = np.arange(0,k,1)
    for x in k_list:
        if x==0 or x==1:
            # это можно написать лучше. может распаковать результат выполнения функции в переменные?
            rmse_list.append(train_and_test(selected_features_data,k = x)[0])
            mse_list.append(train_and_test(selected_features_data,k=x)[1])
            mae_list.append(train_and_test(selected_features_data,k=x)[2])
            mape_list.append(train_and_test(selected_features_data,k=x)[3])
            R_2_list.append(train_and_test(selected_features_data,k=x)[4])
        else:
            rmse_list.append(train_and_test(selected_features_data,k = x)[0])
            mse_list.append(train_and_test(selected_features_data,k = x)[1])
            mae_list.append(train_and_test(selected_features_data,k = x)[2])
            mape_list.append(train_and_test(selected_features_data,k = x)[3])
            R_2_list.append(train_and_test(selected_features_data,k = x)[4])

    return rmse_list,mse_list, mae_list, mape_list, R_2_list

def metrics_output(rmse_list,mse_list, mae_list, mape_list, R_2_list):
    '''получает списки с метриками для всех разбиений от 0 до k. Находит среднее значение каждой метрики,
    выводит значения с некоторыми выводами и комментариями'''

    mse_list = list(deepflatten(mse_list,1))
    print(f'{round(np.mean(np.absolute(mse_list)))} - средний показатель MSE по разбиениям от 0 до {k}.')
    # print('Вывод: конечно огромная ошибка. Возможно является таковой из-за выбросов. Я так и не исследовал выбросы!')

    mae_list = list(deepflatten(mae_list,1))
    print(f'{round(np.mean(np.absolute(mae_list)))} - средний показатель MAE по разбиениям от 0 до {k}.')
    # print('Вывод: средняя абсолютная ошибка при k=6 - 19294 не велика. Это хорошо. при k=10 - 19061 ')

    mape_list = list(deepflatten(mape_list,1))
    print(f'{round(np.mean(np.absolute(mape_list)),2)*100}% - MAPE средняя асболютная ошибка в процентах')
    # print('Средняя абсолютная ошибка в процентах должна быть <10% самый лучший сектор.12 считаю хорошей.')

    R_2_list = list(deepflatten(R_2_list,1))
    print(f'{round(np.mean(np.absolute(R_2_list)),2)} - R^2 коэффициент детерминации')
    print(f'Должен близиться к 1.0')

    mean_sell_price = round(np.mean(final_test_data['SalePrice']))
    print(f'{mean_sell_price}$ - средняя цена продажи ("SalePrice") дома по Генеральной совокупности')
    print(f'{np.mean(rmse_list)} - cредняя всех RMSE (корней среднеквадратичной ошибки) по разбиениям от 0 до {k}')
    print('Тут важно решить: берем среднюю как более честную по точности или берем лучшую (минимальную) потому что'
          'мы же смогли типо добиться наибольшей точности по k-тому количеству разбиений?')
    # print(f'{round(min(rmse_list))}$ - наименьшее RMSE - отклонение модели по цене продажи ("SalePrice") ')
    # print(f'{round(min(rmse_list)/mean_sell_price*100)}% - процент ошибки RMSE от средней цене продажи ("SalePrice")')
    print('Хочу процент ошибки < 15%')


# rmse_list,mse_list, mae_list, mape_list, R_2_list = metrics_lists(k)
# metrics_output(rmse_list,mse_list, mae_list, mape_list, R_2_list)

'''построим простой график, где для каждого k будет своя точность модели, и поймем как это выглядит'''
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# k_list = np.arange(0,10,1) #плохо что генератор здесь и в самом цикле где считается точность разные.
# fig, ax = plt.subplots(figsize=(10,6))
# ax.set(facecolor = 'white')                 #покрасили область Axes в красный
# fig.set(facecolor = 'green')                #покрасили область Figure в зеленый
# lines = plt.plot(k_list,rmse_list,'--b')
# plt.grid()
# # # plt.plot()
# plt.setp(lines,linestyle='-.')
# ax.xaxis.set_major_locator(ticker.MultipleLocator(1)) # интервал основных делений х = 1
# ax.set_xlim(xmin=0, xmax=25)
# ax.set_ylim(ymin=30000, ymax=33000)
# plt.xlabel('k/cv (число блоков разбиения ГС)')
# plt.ylabel('rmse (корень среднеквадратической ошибки) по данному k/cv')
# plt.title('График зависимости величины ошибки (rmse) от числа блоков разбиения (k/cv) ')
# plt.show()

'''получился хороший график. наблюдаем самую высокую точность при k=8. попробуем увеличить число блоков с 10 до 15.
у меня получаются разные точности. почему? random state в cross_val_score не установлен. или сам механизм 
разбиения на блоки предполагает всегда разные варианты - вроде нет..
где может устанавливаться random_state(случайное состояние) еще я называл это посев для генератора псевдослучайных чисел
seed for random number generator:
sample() - установлен 42. я не знаю почему 42. так делал анатолий карпов. можно написать и 1 например
Kfold - установим 42. Причем обязательно должен передаваться параметр shuffle=True. иначе мы установили посев ни для чего.
cross_val_score - здесь нет параметра random_state. точность посева получается достигается через Kfold для cross_val_score
linerarregression - глупо было ожидать здесь random_state. для чего он нужен. подбор параметров идет чисто математически.
по методу наименьших квадратов, если я не ошибаюсь.
Вывод по графику. наблюдается уменьшение ошибки при возрастающем k. на каком числе остановиться? кажется что эксперимент
с k=25 это уже перебор. думаю надо остановиться на 10 или 16..и увеличивать точность дальше другими способами.
'''

# Построй график предсказания и фактические значения! - придется заново написать вручную или как то встроить в конвейер
'''хочу ли я заморочиться и построить два графика на одной области, чтобы их рядом показывало?
для начала давай по простому'''

# вычисляй НЕ все новые метрики.придется построить график фактических и предсказанных значений чтобы проанализировать MSE
print('Predictions')
pred = train_and_test(selected_features_data,k = 0)[5]
# print(train_and_test(selected_features_data,k = 0)[5])
print('Actual value')
actual = train_and_test(selected_features_data,k = 0)[6]
# print(train_and_test(selected_features_data,k = 0)[6])

'''у нас не правильный график, нам нужен не линейный график а. у нас есть только одна координата для каждого массива'''
x = range(0,1469)
fig, ax = plt.subplots(figsize=(10,6))
ax.set(facecolor = 'white')                 #покрасили область Axes в красный
fig.set(facecolor = 'green')                #покрасили область Figure в зеленый
plt.grid()
plt.plot(x,actual,'--b')
# plt.plot(x,pred,'--r')
# plt.plot()
# plt.setp(lines,linestyle='-.')
# ax.xaxis.set_major_locator(ticker.MultipleLocator(1)) # интервал основных делений х = 1
# ax.set_xlim(xmin=0, xmax=25)
# ax.set_ylim(ymin=30000, ymax=33000)
plt.xlabel('Предсказанные значения цены продажи по y_test')
plt.ylabel('Фактические значения цены продажи по y_test')
plt.title('График отклонения предсказанных значений от фактических при k=0')
plt.show()
'''исходя из этого графика видно что буквально пару раз предсказания ушли за 600 и 800 тыс доларов. сильно оторвавшись 
от актуальных значений. 
Но еще видно что большинство актуальных значений помещается в коридор 0 - 400 тыс долларов. Вопрос
может быть выкинуть из фрейма значения более 400 или для начал 600 тыс долларов.'''
