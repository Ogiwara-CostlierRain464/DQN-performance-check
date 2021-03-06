import time
from environment import Environment
from guppy import hpy

before = time.time()
h = hpy()

cart_pole = Environment()
cart_pole.run()

after = time.time()

print("It just took {0}.".format(after - before))

heap = h.heap()
for _ in range(0, 5):
    print(heap)
    heap = heap.more



# region recode

# 11.40
# 10.19
# 24.41
# 4.68
# 19.19

# 14.06
# 25.85
# 6.11 : 27155708
# 4.08 : 27155722
# 6.14 : 27155842

# 17.37 : 27155856 1296 * 2 table
# 4.39 : 27155941
# 15.89 : 27155906
# 6.39 :
# 19.06

# 6.59 : (str: 7864901, tuple: 4244144, bytes: 1861979, type: 1801528, function: 1629504, dict (no owner): 1287088)
# numpy.ndarray: 28110
# 31333 (15.88)
# 6680 (3.71)
# 11263 (5.77)
# 27985 (14.88)
# 18634 (9.60)
# 22012 (11.57)
# 8225 (4.34)
# 25213 (12.90)
# 15654 (8.15)

# 1,795.14 step per 1s.
# VARIANCE 51.219250666666674
# mean: 12.61

# メモリ分割数を増やしてみる
# 6^4 * 2 -> 12^4 * 2
# 28110 -> 414720
# 27.56 : 27469301 (str: 7864901, tuple: 4244144, bytes: 1861979, type: 1801528, function: 1629504, dict (no owner): 1287088)

# 57443(29.89)
# 29787(15.16)
# mean: 22.52 var: 54.24
# FAIL
# FAIL
# FAIL
# FAIL
# FAIL
# FAIL
# FAIL
# ...



# region NN

# 26.34 : 31325620
# 30.94 : 31325798
# 29.53 : 31325968
# 12.29 : 31,173,866
# 25.85 :

# 13.28
# 30.86 :
# 17.62
# 20.54
# 19.04

# 27.42
# 10.20
# 24.11
# 11.36
# 17.62

# VARIANCE 49.55854222222222
# mean: 21.13

# つまり、
# 時間のばらつき具合は変わらない(どちらもかなりばらつく)
# メモリ使用量も安定的(高々 1MB) 27MB, 31MB
# Tableの場合は順調に比例？

# step評価
# FAIL
# 13709 (17.93)
# 13768 (17.74)
# 6103 (7.91)
# 12072 (17.17)
# 27685 (35.77)
# 18330 (23.46)
# 5178 (6.82)
# 15215 (19.73)
# 19727 (25.14)
# FAIL

# ^ mean: 19.07 var: 68.52


# unit数を16に

# 18.42 : 31329361
# 25.00
# 23
# 24
# 26
# 28
# 31
# 36
# 29


# 今度はstep数で評価?
# 7179 (9.22)
# 14337 (18.10)
# 22932 (28.84)
# 41792 (52.25)
# FAIL
# 22200 (28.11)
# 18634 (23.74)
# 27991 35.59

# ^ mean: 27.97 var: 159.318

# endregion

# endregion
