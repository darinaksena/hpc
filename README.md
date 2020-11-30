# HPC L1
## System
Google Colab: https://colab.research.google.com/drive/151805XTDg--dgHb3-AXJCpnWaqRhop_2#scrollTo=ojGuEt8MpJhA
## Experiment
В ходе работы был написан код для двух экспериментов - сравнения ускорения для умножения матриц при помощи GPU, в том числе и с разделяемой памятью, относительно параллельного алгоритма перемножения матриц.
Алгоритм с разделяемой памятью реализует блочный алгоритм перемножения матриц, когда данные загружаются в разделяемую память частями, после чего промежуточные результаты суммируются. Второй алгоритм с более простой реализацией производит многократное копирование данных для нити и перемножает матрицы стандартным способом.

Для последовательной реализации была использована функция dot библиотеки numpy, которая оптимизирует работу с массивами и производит вычисления быстрее, чем любая собственная последовательная реализация (от которой целесообразно отказаться ввиду низкой скорости), но даже оптимизация нпри больших размерностях матриц работает медленнее, чем параллельное умножение, о чем можно судить по результатам эксперимента.

Так же из результатов эксперимента видно, что при относительно небольших размерностях матриц параллельные вычисления не дают выигрыша во времени работы за счет передачи данных.

Для эксперимента были использованы матрицы, размерности которых кратны 32, чтобы загрузка блоков GPU была максимальной.

## Results
### Device memory
|Matrix size|CPU time, s|GPU time, s|Speedup|
|-----------|--------|--------|-------|
|128|0.00021266937255859375| 0.04872474670410156|0.004364709658730594|
| 256|0.0011779308319091798|0.0004307746887207031| 2.734447642240425|
| 512|0.008749437332153321|0.0005839347839355468|14.98358647721705|
|1024|0.058165359497070315| 0.0005691051483154297| 102.20494344365312|
|2048|0.44664726257324217| 0.0004214763641357422| 1059.7207828939925|


### Shared memory
|Matrix size|CPU time, s|GPU time, s|Speedup|
|-----------|--------|--------|-------|
|128|0.00020213127136230468| 0.04845128059387207| 0.00417184579818659|
|256| 0.0011984825134277344| 0.00040311813354492186| 2.973030518097942|
| 512| 0.009438800811767577 | 0.0005093574523925781| 18.530799475753604|
|1024| 0.0582852840423584| 0.0005350112915039062 | 108.9421568627451|
| 2048 | 0.4525922298431396 | 0.0004405975341796875 | 1027.2237012987014 |


# HPC L2
## Experiment
В ходе работы был написан код для реализации генетического алгоритма на C++/CUDA. 
Алгоритм:
  Создаем первое поколение на CPU.
  Затем на GPU происходит кроссинговер и мутация. 
  Далее управление возвращается CPU, где вычисляется показатель фитнесс и происходит отбор.
  Цикл повторяется до тех пор, пока не выполнится одно из условий: достигнуто максимальное число поколений или не наблюдается улучшения точности (достигнуто число повторов).

В качестве функции для аппроксимации был выбран sin(x).
На вход программмы подается кол-во точек функции sin(x), кол-во элементов в поколении, среднее и дисперсия для мутации, кол-во поколений и кол-во поколений с одинаковыми результатами (критерии прекращения работы алгоритма).
