import copy

a = [1, 2,[1]]
b = copy.copy(a)  # 浅拷贝

# print(a is b)  # False，不是同一个对象
# print(a[1] is b[1])  # True，内部列表是同一个对象

b[2].append(2)  # 修改嵌套列表
print(a)  # [1, 2, [3, 4, 5]]，原列表也受到影响
print(b)  # [1, 2, [3, 4, 5]]
