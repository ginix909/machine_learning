'''
Модель парной линейной регрессии является частным случаем модели многомерной регрессии.
Её исследование представляет самостоятельный интерес, так как она имеет многие характерные свойства общих многомерных
моделей, но более наглядна и проста для изучения.
Задача - линейного регрессионного анализа - по имеющимся двум масивам данных получить наилучшие оценки b0 , b1
чтобы построить эмперическое уравнение регрессии y = b0 + b1 * x


1.Постройте поле корреляции и сформулируйте гипотезу о форме связи.
2.Рассчитайте параметры выборочного уравнения линейной регрессии с помощью МНК.
3.Оцените тесноту связи с помощью показателей корреляции (выборочный коэффициент корреляции) и детерминации.
4.Используя критерий Стьюдента оцените статистическую значимость коэффициентов регрессии и корреляции.
5.Постройте интервальные оценки параметров регрессии. Проверьте, согласуются ли полученные результаты с выводами,
полученными в предыдущем пункте.
6.Постройте таблицу дисперсионного анализа для оценки значимости уравнения в целом.
7.С помощью теста Гольдфельда – Квандта исследуйте гетероскедастичность остатков. Сделайте выводы.
8.В случае пригодности линейной модели рассчитайте прогнозное значение результата,
если значение фактора увеличится на 5% от его среднего уровня.
Определите доверительный интервал прогноза для уровня значимости =0,05.
9.Оцените полученные результаты, проинтерпретируйте полученное уравнение регрессии.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.stats as stats

df = pd.read_csv('state.csv')
x_name = ('hs_grad','уровень образования')
y_name = ('poverty', 'уровень бедности')

ax = df.plot.scatter(x=x_name[0], y=y_name[0])
ax.set_xlabel('Среднее образование (%)')
ax.set_ylabel('Бедность (%)')
ax.set_title('Связь бедности и уровня образования')
ax.grid(True)
# ax.legend(loc='upper center')
ax.set_xlim(xmin=76, xmax=93)
ax.set_ylim(ymin=5, ymax=20)

#  Устанавливаем интервал основных делений:
ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
#  Устанавливаем интервал вспомогательных делений:
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

#  Тоже самое проделываем с делениями на оси "y":
ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

# plt.show()

x_streak = round((pd.Series.mean(df['hs_grad'])),1)
y_streak = round((pd.Series.mean(df['poverty'])),1)

N_x = pd.Series.count(df['hs_grad'])
N_y = pd.Series.count(df['poverty'])

hs_grad = pd.Series(df['hs_grad'])
poverty= pd.Series(df['poverty'])
common_sum = 0

for x,y in zip(hs_grad,poverty):
    common_sum += (x-x_streak)*(y-y_streak)

common_sum = round(common_sum,2)
cov = round((common_sum/(N_x-1)),1)
dispersion_x = 0
dispersion_y = 0
for x in hs_grad:
    dispersion_x += (x-x_streak)**2

for y in poverty:
    dispersion_y += (y-y_streak)**2

dispersion_x = dispersion_x/N_x
dispersion_y = dispersion_y/N_y
sd_x = round(dispersion_x**0.5,1)
sd_y = round(dispersion_y**0.5,1)
correlation = round((cov / (sd_x*sd_y)),2)


'''
Далее построим регрессионную прямую (чтобы вывести уравнение) с помощью метода наименьших квадратов
МНК - метод нахождения оптимальных параметров линейной регрессии, таких что сумма квадратов ощибок (остатков)
была минимальна.
Параметры в даннос случае это b0 , b1

b1 = (SDy / SDx) * rxy 
b0 = Ystreak - b1 * Xstreak
'''
b1 = round(((sd_y / sd_x) * correlation),3)
b0 = round((y_streak-b1*x_streak),3)

# создаем массив точек для линии регресии, подставляя в уравнение регрессии все точки массива Х.
y_list = [ round((b0+b1*x),2) for x in pd.Series(df['hs_grad'])]

# строим линию регрессии
plt.plot(pd.Series(df['hs_grad']), y_list)
# plt.show()

R2 = correlation**2

m=2
degrees_of_freedom = N_x - m
# print(f'Число степеней свободы по {m} группам: {degrees_of_freedom}')

# рассчет t - значения, вероятности p и стандартной ошибки для X параметра
tx = correlation* np.sqrt(degrees_of_freedom / ((1.0 - correlation)*(1.0 + correlation)))
sterrestx = np.sqrt((1 - correlation**2) * dispersion_y / dispersion_x/ degrees_of_freedom)
px = 2 * stats.t.sf(np.abs(tx), degrees_of_freedom)

# Находим сумму квадратов по параметру X
s = [i**2 for i in hs_grad]
sterresty = (sterrestx**2/51*sum(s))**0.5
ty = b0/sterresty
py = 2 * stats.t.sf(np.abs(ty), degrees_of_freedom)
t_critical = 2.011

# рассчет F-статистики подсмотренно тут:
# https://www.chem-astu.ru/science/reference/fischer.html
F = (correlation**2/(1-correlation**2))*(N_x-m-1)
p_val = stats.f.sf(F, 1, degrees_of_freedom)
# print(f'F-statistic(1,{degrees_of_freedom}) = {round(F,2)}, p-value = {p_val}')

alpha = 0.05
F_critical = 3.2


print('Последовательный вывод')
print(f'Мы строим модель парной линейной регрессии вида y= b0 + b1 * x , '
      f'где y это {y_name[1]} - зависимая переменная, ')
print(f'х это {x_name[1]} - независимая переменная')
correlation_type = 'Корреляционная связь между переменными '
if correlation>0:
    vector = 'прямая и '
    correlation_type += vector
elif correlation ==0:
    vector = 'отсутсвует,'
    correlation_type += vector
elif correlation < 0:
    vector = 'обратная и '
    correlation_type += vector
if 0.1 <= abs(correlation) < 0.3:
    correlation_type += 'слабая'
elif 0.3 <= abs(correlation) <0.5:
    correlation_type += 'умеренная'
elif 0.5 <= abs(correlation) < 0.7:
    correlation_type += 'заметная'
elif 0.7<= abs(correlation) <0.9:
    correlation_type += 'высокая'
elif 0.9 <= abs(correlation) <=0.99:
    correlation_type += 'весьма высокая'


# if b1>0:
#     print('Взаимосвязь исследуемых величин положительная')
#     print(f'С каждым процентом увеличения {x_name}, мы ожидаем что {y_name} будет увеличиваться на {b1}% ')
# print('Взаимосвязь исследуемых величин отрицательная')
# print(f'С каждым процентом увеличения {x_name}, мы ожидаем что {y_name} будет уменьшаться на {abs(b1)}% ')
# print(f'Практически {int(R2*100)}% изменчивости показателя {x_name} объясняется нашей моделью')

print(f'1.Мы построили поле корреляции в виде графика и сформулировали гипотезу о форме связи')
# plt.show()
print(f'Гипотеза - имеется ли взаимосвязь между переменными, прямая/обратная, сильная/слабая')
print(f'описывается прямой или другой линией ')

print(f'2.Рассчитали параметры выборочного уравнения линейной регрессии с помощью МНК')
print(f'X штрих (средняя х): {x_streak}, Y штрих (средняя y): {y_streak}')
print(f'N (количество наблюдений) х: {N_x}, N (количество наблюдений) y: {N_y}')
print(f'Standart deviation (стандартное отклонение) x:{sd_x}')
print(f'Standart deviation (стандартное отклонение) y:{sd_y}')
print(f'Covariation: {cov}, Correlation: {correlation}')
print(f'Коэффициент b1 (слоуп/угол наклона): {b1}')
print(f'Коэффициент b0 (интерсепт/точка пересечения линии регрессии с ОУ): {b0}')
print(f'Уравнение линейной регрессии имеет вид y = {b0} + {b1} * x')

print('3.Оцените тесноту связи с помощью показателей корреляции (выборочный коэффициент корреляции) и детерминации.')
print(f'Коэффициент детерминации принимает значения от 0 до 1. В случае качественной модели стремится к 1.')
print(f'Коэффициент детерминации: {R2} означает, что {abs(R2*100)}% вариации у:"{y_name[1]}" объясняется вариацией '
      f'фактора х:"{x_name[1]}"')
print(f'а что {100 - abs(R2*100)}% - действием других факторов, не включённых в модель.')
print(f'Коэффициент корреляции: {correlation}')
print(f'{correlation_type}')

print(f'4.Используя критерий Стьюдента оцените статистическую значимость коэффициентов регрессии и корреляции.')
# print(f't-значение (критерий стьюдента) b0 ({y_name[0]}):{round(ty,2)}')
print(f't-значение (критерий стьюдента) b1 ({x_name[0]}):{round(tx,2)}')
print(f'H0 - нулевая гипотеза утверждает что если значения не коррелируют, r=0, то b1=0 и линия регрессии '
      f'параллельна оси ОХ ')
print(f'H1 - альтернативная гипотеза утверждает, что связь есть и b1!=0')
print('t-критерий проверяет гипотезу о том, что коэффициент b1 ответственный за угол наклона отличен от 0.')
print(f'Нужно сравнить табличное значение t-критерия с фактическим')
print(f'Так как уровень значимости а=0.05, число степеней свобод df = N-2 т.е. {N_x-2}')
# print(f'Значит табличное значение t-критерия = {t_critical}')
print(f'Фактическое значение p-уровня значимости t-критерия = {round(px,4)} ')
p_critical= 0.05
if px < p_critical:
    print(f'Фактическое значение p-уровня значимости t_x = {round(px,4)} меньше допустимого p={p_critical} значит')
    print('НУЛЕВАЯ ГИПОТЕЗА ОТКЛОНЯЕТСЯ')
    print('Из этого следует, что Коэффициент корреляции СТАТИСТИЧЕСКИ ЗНАЧИМ')
elif px > p_critical:
    print(f'Фактическое значение p-уровня значимости t-критерия {round(px,4)} выше допустимого, значит')
    print('НУЛЕВУЮ ГИПОТЕЗУ ОТКЛОНИТЬ НЕЛЬЗЯ!!!')

print(f'Для параметра y (b/b0) критерий проверки значимости имеет вид t_b0 = b/st_err_regr')
print(f' где b - оценка коэф регрессии, st_err_regr - стандартная ошибка коэф регрессии.')
print(f't-уровень значимости y (b0): {round(ty,2)}')
if py < p_critical:
    print(f'Фактическое значение p-уровня значимости t_y = {round(py,2)} меньше допустимого p={p_critical}, значит')
    print('НУЛЕВАЯ ГИПОТЕЗА ОТКЛОНЯЕТСЯ')
    print('Из этого следует, что Коэффициент регрессии СТАТИСТИЧЕСКИ ЗНАЧИМ')
elif py > p_critical:
    print(f'Фактическое значение p-уровня значимости t_y-критерия {round(py,2)} выше допустимого, значит')
    print('НУЛЕВУЮ ГИПОТЕЗУ ОТКЛОНИТЬ НЕЛЬЗЯ!!!')

print('5.Построим интервальные оценки параметров регрессии. Проверим, согласуются ли полученные результаты с выводами,'
      'полученными в предыдущем пункте.')
print('Определим доверительный интервал коээффициента b1 и проверим гипотезу о равенстве нулю коэффициента '
      'направления прямой парной линейной регрессии.')

left_conf_interval_x = x_streak - 1.96 * sd_x / N_x ** 0.5
right_conf_interval_x = x_streak + 1.96 * sd_x / N_x ** 0.5
print(f'Если бы многократно повторяли эксперимент, то все выборочные средние распределились бы нормальным образом')
print('вокруг среднего генеральной совокупности(то что мы ищем) со стандартной ошибкой среднего')
print('ст.отклонением) = sdx/ n**0.5. И 95% выборочных средних по свойству номрального распределения нашей средней')
print('в Генеральной совокупности лежали бы в диапазоне +-1,96сигм(стандартных отклонений или в нашем случае (se)')
print('Так как наша выборочная средняя с вероятностью в 95% нахоится в этом 95% интервале для ГС,')
print(f'то средняя ГС точно находится в интервале от {round(left_conf_interval_x, 2)} до {round(right_conf_interval_x, 2)}')
print(f'Или доверительный интервал для х: {round(left_conf_interval_x, 2)} до {round(right_conf_interval_x, 2)}')


if not left_conf_interval_x<0<right_conf_interval_x:
    print(f'Так как гипотетическое значение коэффициента b1 - нуль - не принадлежит доверительному интервалу, ')
    print('с вероятностью 95% можем ОТВЕРГНУТЬ НУЛЕВУЮ ГИПОТЕЗУ и принять альтернативную гипотезу, то есть считать,')
    print(f' что зависимая переменная Y:{y_name[1]} линейно зависит от независимой переменной X:{x_name[1]}.')
else:
    print(f'Так как гипотетическое значение коэффициента b1 - нуль - принадлежит доверительному интервалу, ')
    print(f'НЕЛЬЗЯ ОТВЕРГНУТЬ НУЛЕВУЮ ГИПОТЕЗУ о равенстве коэффициента b1 нулю, что означает,')
    print(f' что переменная Y:{y_name[1]} не зависит от переменной X:{x_name[1]}.')


# y_regr = b0 + b1* x
y_regr_list = [(b0 + b1* x) for x in hs_grad]
e_list = []
for y,y_regr in zip(poverty,y_regr_list):
    e_list.append((y-y_regr)**2)
RSS = sum(e_list) #остаточная сумма квадратов отклонений SSE

# остатки - разности между реальными значениями зависимой переменной и значениями, оценёнными уравнением линейной регрессии.
print(f'Сумма квадратов остатков RSS = {RSS}. В случае качественной модели стремится к нулю.')
SEE = (RSS/N_x-m-1)**0.5
print(f'Стандартная ошибка регрессии (SEE) измеряет величину квадрата ошибки, приходящейся на одну степень свободы модели:')
print(f'SEE = {round(SEE,2)}. Чем меньше значение SEE, тем качественнее модель.')

s_b1 = SEE / (sum([(x-x_streak)**2 for x in hs_grad]))**0.5
print(f'Стандартная погрешность коэффициента направления прямой линейной регресии b1 - s_b1 = {round(s_b1,4)}')
t_value = 2.01
print(f'Так как t-знаение = {t_value} для р-уровня значимости 0.05 и df=49 ')

print(f'6.Построим таблицу дисперсионного анализа для оценки значимости уравнения в целом.')
# TSS = SUM(yi - y_streak) - общая сумма квадратов отклонений
# SSR = SUM(y_regr - y_streak) - регрессионная сумма квадратов отклонений, сумма кв из-за регрессии
# SSE = SUM(yi - y_regr) - остаточная сумма квадратов отклонений, сумма кв ошибок прогнозирования
TSS = sum([(y - y_streak)**2 for y in poverty])
SSR = sum([(y_reg -y_streak)**2 for y_reg in y_regr_list])
SSE = sum([(y-y_regr)**2 for y,y_regr in zip(poverty,y_regr_list)])
# print(f'TSS= {TSS}')
# print(f'SSR= {SSR}')
# print(f'SSE= {SSE}')
if TSS==SSR + SSE :
    print(f'TSS = SSR + SSE')
else:
    print('TSS!= SSR + SSE')

F_fact = (SSR / SSE ) * ((N_x-m-1)/m)
# print(f'F_фактическое значение критерия Фишера: {round(F_fact,2)}')
F_table = 3.2
# print(f'F_табличное значение критерия Фишера: {F_table}')
if F_fact>F_table:
    print(f'F-критерий фактический:{round(F_fact,2)} больше чем F-критерий табличный:{F_table}.Следовательно необходимо')
    print('ОТКЛОНИТЬ H0 - нулевую гипотезу  и сделать вывод о статистической значимости уравнения регрессии в целом')
    print('и значения R**2, так как они статистически надежны и сформировались под систематическим действием неслучайных причин.')
    print('Также это значит, что объясненная дисперсия существенно больше,чем необъясненная.')

else:
    print(f'F-критерий фактический:{round(F_fact,2)} меньше чем F-критерий табличный:{F_table}. Следовательно')
    print('НЕЛЬЗЯ ОТКЛОНИТЬ H0 - нулевую гипотезу  и сделать вывод о статистической значимости уравнения регрессии в целом')
    print('и значения R**2, так как они статистически надежны и сформировались под систематическим действием неслучайных причин.')

# создаем таблицу дисперсионного анализа
dispersion_table = pd.DataFrame(data = {
    'df':[N_x-1,m,N_x-m-1],
    'Сумма квадратов отклонений, SS':[TSS,SSR,SSE],
    'Дисперсия на степень свободы, MS':['-',SSR/m,SSE/(N_x-m-1)],
    'F-фактическое':[F_fact,'',''],
    'F-табличное':[F_table,'','']},
    index = ['Общая вариация результата, y','Факторная вариация результата, y', 'Остаточная вариация результата, y'])
pd.options.display.width= None
pd.options.display.max_columns= None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)
print(dispersion_table)

print('7.С помощью теста Гольдфельда – Квандта исследуйте гетероскедастичность остатков. Сделайте выводы.')
# Проверить свои вычисления встроенными методами статистики желательно одного пакета, скорее всего scipy
# этот пункт я еще должен доделать. позже
