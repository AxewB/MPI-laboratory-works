## Laboratory No1 (Shared variant). Запуск параллельной программы на различном числе одновременно работающих процессов, упорядочение вывода результатов.

1. Написать параллельную программу MPI, где каждый процесс определяет свой ранг `MPI_Comm_rank (MPI_COMM_WORLD, &ProcRank);`, после чего действия в программе разделяются. Все процессы, кроме процесса с рангом 0 else, передают значение своего ранга нулевому процессу `MPI_Send (&ProcRank, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);`. Процесс с рангом 0 `if ( ProcRank == 0 ){...}` сначала печатает значение своего ранга `printf ("\n Hello from process %3d", ProcRank);`, а далее последовательно принимает сообщения с рангами процессов `MPI_Recv(&RecvRank, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &Status);` и также печатает их значения `printf("\n Hello from process %3d", RecvRank); `. При этом важно отметить, что порядок приема сообщений заранее не определен и зависит от условий выполнения параллельной программы (более того, этот порядок может изменяться от запуска к запуску).
2. Запустить программу на 1,2 … N процессах несколько раз.
3. Проанализировать порядок вывода сообщений на экран. Вывести правило, определяющее порядок вывода сообщений.
4. Построить график времени работы программы в зависимости от числа запущенных процессов от 1 до 16. Размер шага – например, 4.
5. Построить график ускорения/замедления работы программы.
6. Модифицировать программу таким образом, чтобы порядок вывода сообщений на экран соответствовал номеру соответствующего процесса.
7. Нарисовать сеть Петри для двух вариантов MPI программы.

## Laboratory No2 (Variant 12). Использование функций обменда данными "Точка-точка" в библиотеке MPI

**Умножение векторов.** Напишите программу, реализующую скалярное произведение двух распределенных между процессами векторов (без использования коллективных операций)

## Laboratory No3 (Variant 3). Использование аргументов-джокеров

**Имитация топологии "звезда" (процесс с номером 0 реализует функцию центрального узла).** Процессы в случайном порядке генерируют пакеты, состоящие из адресной и информационной части и передают их в процесс 0. Маршрутная часть пакета содержит номер процесса-адресата. Процесс 0 переадресовывает пакет адресату. Адресат отчитывается перед процессом 0 в получении. Процесс 0 информирует процесс-источник об успешной доставке.

## Laboratory No4 (Variant 5). Коллективные операции

В каждом процессе дан набор из K чисел, где K — количество процессов. Используя функцию MPI_Alltoall, переслать в каждый процесс по одному числу из всех наборов: в процесс 0 — первые числа из наборов, в процесс 1 — вторые числа, и т. д. В каждом процессе вывести числа в порядке возрастания рангов переславших их процессов (включая число, полученное из этого же процесса).

## Laboratory No5 (Variant 9). Группы процессов и коммуникаторы

В каждом процессе дано целое число N, которое может принимать два значения: 0 и 1 (имеется хотя бы один процесс с N = 1). Кроме того, в каждом процессе с N = 1 дано вещественное число A. Используя функцию MPI_Comm_split и одну коллективную операцию редукции, найти сумму всех исходных чисел A и вывести ее во всех процессах с N = 1.
Указание. При вызове функции MPI_Comm_split в процессах, которые не ¬¬требуется включать в новый коммуникатор, в качестве параметра color следует указывать константу MPI_UNDEFINED.

## Laboratory No6 (Variant 8). Виртуальные топологии

Количество процессов K равно 8 или 12, в каждом процессе дано вещественное число. Определить для всех процессов декартову топологию в виде трехмерной решетки размера 2 × 2 × K/4 (порядок нумерации процессов оставить прежним). Интерпретируя полученную решетку как K/4 матриц размера 2 × 2 (в одну матрицу входят процессы с одинаковой третьей координатой), расщепить эту решетку на K/4 указанных матриц. Используя одну коллективную операцию редукции, для каждой из полученных матриц найти сумму исходных чисел и вывести найденные суммы в каждом процессе соответствующей матрицы.

## Laboratory No7 (Variant 3). Умножение матриц

Блочный алгоритм Кэннона
