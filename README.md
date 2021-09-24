# Приложение для интерактивного отображения данных на карте

Здравствуйте, это приложение для интерактивного отображения данных на карте Алтайского края, сделанное по просьюе
Алтайского государственного университета

# Онлайн пример работы с набором данных о распространении ВИЧ в Алтайском крае в 2019 году

[https://dash-map-app.herokuapp.com/](https://dash-map-app.herokuapp.com/)

# Инструкция по установке

1. [Установите Python >3.8](https://www.python.org/ftp/python/3.9.7/python-3.9.7-amd64.exe) (ссылка для windows x64)
2. Скачайте проект [здесь](https://github.com/egorserdyuk/Diploma/archive/refs/heads/side-project.zip) или через
   git [https://github.com/egorserdyuk/Diploma.git](https://github.com/egorserdyuk/Diploma.git)
3. Распакуйте в папку и зайдите в нее через командную строку/терминал
4. Установите виртуальную среду командой ```python -m venv venv```
    1. Войдите в нее командой ```.\venv\Scripts\activate```
5. Установите все прилагающиеся библиотеки командой ```pip install -r requirements.txt```
6. Запускайте сервис после установки всех библиотек через ```python main.py```

# Смена набора данных

Чтобы сменить набор данных для отображения, нужно использовать сторонний путь - редактировать название файла данных,
лежащего в папке ```data```, в коде:

```python
df = pd.read_csv("data/HIV.csv", dtype={"County": str})  # Заменяем название файла, который подгружаем в переменной df
```

# Разметка

Как описывалось ранее, в дипломной работе, есть два обязательных столбца ```County и Data```, где ```County``` содержит
информацию о районе на английском языке, а ```Data``` значения, которые будут отображаться.

Пример:

| County | Data |
| --- | --- |
| Suetsky | 0.0 |
| Kosikhinsky | 69.1 |

# База названий районов

|Район|County|id|
| --- | ---- |---|
|Krutikhinsky|Крутихинский|1|
|Pankrushihinsky|Панкрушихинский|2|
|Burlinsky|Бурлинский|3|
|Nemetsky|Немецкий|4|
|Khabarsky|Хабарский|5|
|Slavgorod|Славгородский|6|
|Tabunsky|Табунский|7|
|Kulundinsky|Кулундинский|8|
|Uglovsky|Угловский|9|
|Mikhaelovsky|Михайловский|10|
|Klyuchevsky|Ключевский|11|
|Rubtsovsky|Рубцовский|12|
|Loktevsky|Локтевский|13|
|Tretyakovsky|Третьяковский|14|
|Zmeinogorsky|Змеиногорский|15|
|Charyshsky|Чарышский|16|
|Soloneshensky|Солонешский|17|
|Altaysky|Алтайский|18|
|Sovetsky|Советский|19|
|Krasnogorsky|Красногорский|20|
|Soltonsky|Солтонский|21|
|Zalesovsky|Залесовский|22|
|Talmensky|Тальменский|23|
|Kamensky|Каменский|24|
|Shelabolikhinsky|Шелоболихинский|25|
|Zarinsky|Заринский|26|
|Togulsky|Тогульский|27|
|Eltsovsky|Ельцовский|28|
|Kytmanovsky|Кутмановский|29|
|Biysk|г. Бийск|30|
|Biysky|Бийский|31|
|Tselinny|Целинный|32|
|Barnaul|г. Барнаул|33|
|Kalmansky|Калманский|34|
|Pavlovsky|Павловский|35|
|Kosikhinsky|Косихинский|36|
|Tyumentsevsky|Тюменцевский|37|
|Rebrikhinsky|Ребрихинский|38|
|Topchikhinsky|Топчихинский|39|
|Troitsky|Троицкий|40|
|Zonalny|Зональный|41|
|Pervomaysky|Первомайский|42|
|Suetsky|Суетский|43|
|Blagoveshensky|Благовещенский|44|
|Zavyalovsky|Завьяловский|45|
|Baevsky|Баевский|46|
|Kuryinsky|Курьинский|47|
|Ust-Pristansky|Усть-Пристанский|48|
|Smolensky|Смоленский|49|
|Bistroistokinsky|Быстроистокский|50|
|Petropavlovsky|Петропавловский|51|
|Pospelikhinsky|Поспелехинский|52|
|Rodinsky|Родинский|53|
|Romanovsky|Романовский|54|
|Mamontovsky|Мамонтовский|55|
|Novoegorievsky|Егорьевский|56|
|Novichikhinsky|Новичихинский|57|
|Volchihinsky|Волчихинский|58|
|Aleysky|Алейский|59|
|Shipunovsky|Шипуновский|60|
|Krasnoshekovsky|Краснощековский|61|
|Ust-Kalmansky|Усть-Калманский|62|