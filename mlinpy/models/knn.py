import numpy as np
from operator import itemgetter

def euclidean(a, b):
    return np.sqrt(np.sum(np.square(a-b)))

class KDNode():
    def __init__(self, split, sample, parent, left, right):
        self.split = split
        self.sample = sample
        self.parent = parent
        self.left = left
        self.right = right

    def is_root(self):
        return not self.parent

    def is_leaf(self):
        return not (self.left or self.right)

    def is_left(self):
        return self.parent and self.parent.left is self

    def is_right(self):
        return self.parent and self.parent.right is self

    def get_sibling(self):
        if self.parent and self.parent.left is self:
            return self.parent.right
        elif self.parent and self.parent.right is self:
            return self.parent.left

def create_kd_tree(data):
    if data is None or data.shape[0] == 0 or data.shape[1] == 0:
        return None
    split = np.argmax(np.var(data, axis=0))
    data = np.array(sorted(data, key=lambda x: x[split]))
    head_index = len(data) // 2
    left = create_kd_tree(data[:head_index])
    right = create_kd_tree(data[head_index+1:])
    head = KDNode(split, data[head_index], None, left, right)
    if left:
        left.parent = head
    if right:
        right.parent = head
    return head

def find_nearest(kd_tree, target):
    if len(kd_tree.sample) != len(target):
        raise ValueError('dim not match!')
    if not kd_tree or target is None:
        return -1
    head = kd_tree
    while not head.is_leaf():
        if target[head.split] <= head.sample[head.split]:
            head = head.left
        else:
            head = head.right
    curr_node = head
    curr_dis = euclidean(target, curr_node.sample)
    while not head.is_root():
        if np.abs(head.parent.sample[head.split] - target[head.parent.split]) < curr_dis:
            sbiling = head.get_sibling()
            if sbiling:
                dis = euclidean(target, sbiling.sample)
                if dis < curr_dis:
                    curr_node = sbiling
                    curr_dis = dis
            dis = euclidean(target, head.parent.sample)
            if dis < curr_dis:
                curr_node = head.parent
                curr_dis = dis
        head = head.parent
    return curr_node

from time import time

data = np.random.randn(1000000, 100)

start = time()
kdtree = create_kd_tree(data)
print(time()-start)

start = time()
x = find_nearest(kdtree, np.random.randn(100,)).sample
print(time()-start)
