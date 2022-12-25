'''
В исходном наборе данных 82 колонки. Минус 1 целевая колонка(переменная), минус колонка порядкового номера записи.
она не нужна так как есть PID - id дома.
В исходном наборе данных 80 признаков.
'Order' -
'PID' -
'MS SubClass' - тип жилого помещения, участвующего в продаже. 16 типов. Этаж, год постройки, тип жилья, тип дизайна
'MS Zoning' - общ классификация. 8 типов. Назначение земли(с.х, коммерческое, жилое, плонтость застройки.
        A Сельское хозяйство
        C Коммерческий
        FV Плавающая деревня Жилой дом
        я промышленный
        RH Жилой район с высокой плотностью застройки
        RL Жилой район с низкой плотностью
        RP Жилой парк низкой плотности
        RM Жилая средней плотности

'Lot Frontage'(Фасад участка) - Линейные футы улицы, соединенной с участком. Возможно длина участка по улице.
'Lot Area'(площадь участка) - Размер участка в квадратных футах.
'Street' - тип подьезда к участку. Гравий/дорожное полотно(асфальтировано)
    Grvl	Гравий
    Pave	Асфальт

'Alley' - тип доступа к собественности через переулок. Возможно "подходные пути". Гравий/асфальт/нет доступа
'Lot Shape'-Форма участка. Регулярная регистрация/Слегка неправильный/Умеренно нерегулярный/Нерегулярный
'Land Contour'-
'Utilities',
'Lot Config',
'Land Slope
'Neighborhood',
'Condition 1',
'Condition 2',
'Bldg Type',
'House Style',
'Overall Qual' - Оценка общая материалов и отделки дома
'Overall Cond',
'Year Built',
'Year Remod/Add',
'Roof Style',
'Roof Matl',
'Exterior 1st',
'Exterior 2nd',
'Mas Vnr Type',
'Mas Vnr Area' - Площадь каменной кладки в квадратных метрах
'Exter Qual',
'Exter Cond',
'Foundation',
'Bsmt Qual',
'Bsmt Cond',
'Bsmt Exposure',
'BsmtFin Type 1',
'BsmtFin SF 1',
'BsmtFin Type 2',
'BsmtFin SF 2',
'Bsmt Unf SF',
'Total Bsmt SF' - общая площадь подвала в квадратных футах.
'Heating',
'Heating QC',
'Central Air',
'Electrical',
'1st Flr SF' - площадь 1-го этажа в кв футах
'2nd Flr SF' -
'Low Qual Fin SF',
'Gr Liv Area' - жилая площадь, кв.футы
'Bsmt Full Bath',
'Bsmt Half Bath',
'Full Bath' -полноценная ванная комната над землей
'Half Bath',
'Bedroom AbvGr',
'Kitchen AbvGr',
'Kitchen Qual',
'TotRms AbvGrd' - total rooms above grade - количество комнат над уровнем земли (без подвала)
'Functional',
'Fireplaces' - количество каминов
'Fireplace Qu' - качество/отделка каминов
'Garage Type',
'Garage Yr Blt',
'Garage Finish',
'Garage Cars',
'Garage Area',
'Garage Qual',
'Garage Cond',
'Paved Drive',
'Wood Deck SF' - Площадь деревянного настила в квадратных метрах
'Open Porch SF' - Площадь открытой веранды в квадратных метрах
'Enclosed Porch',
'3Ssn Porch',
'Screen Porch',
'Pool Area',
'Pool QC',
'Fence',
'Misc Feature',
'Misc Val',
'Mo Sold' - месяц продажи
'Yr Sold' - год продажи
'Sale Type' - тип сделки
'Sale Condition' - условия продажи
'SalePrice - цена продажи - ЦЕЛЕВАЯ ПЕРЕМЕННАЯ

'''
