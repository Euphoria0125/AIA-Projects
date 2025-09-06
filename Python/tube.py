import numpy as np
from collections import deque

def find_path(matrix, start, end):
    # 获取三维空间的大小
    i, j, k = matrix.shape
    
    # 初始化一个三维数组用于记录路径
    path = np.zeros((i, j, k, 3), dtype=int)
    
    # 定义一个队列用于存储待访问的节点
    queue = deque()
    
    # 定义一个列表用于存储已访问的节点
    visited = set()
    
    # 将起始点加入队列
    queue.append(start)
    
    # 开始广度优先搜索
    while queue:
        # 取出队列中的第一个节点
        node = queue.popleft()
        
        # 将节点标记为已访问
        visited.add(node)
        
        # 判断是否到达目标点
        if node == end:
            break
        
        # 获取节点的邻居节点
        neighbors = get_neighbors(node)
        
        # 遍历邻居节点
        for neighbor in neighbors:
            # 判断邻居节点是否已访问
            if neighbor not in visited:
                # 判断邻居节点是否可用
                if matrix[neighbor[0], neighbor[1], neighbor[2]] == 1:
                    # 将邻居节点加入队列
                    queue.append(neighbor)
                    
                    # 更新路径
                    path[neighbor[0], neighbor[1], neighbor[2]] = node
    
    # 回溯路径
    path = backtrack_path(path, start, end)
    
    return path

def get_neighbors(node):
    # 获取一个节点的邻居节点
    i, j, k = node
    
    neighbors = []
    
    if i > 0:
        neighbors.append((i-1, j, k))
    if i < matrix.shape[0]-1:
        neighbors.append((i+1, j, k))
    if j > 0:
        neighbors.append((i, j-1, k))
    if j < matrix.shape[1]-1:
        neighbors.append((i, j+1, k))
    if k > 0:
        neighbors.append((i, j, k-1))
    if k < matrix.shape[2]-1:
        neighbors.append((i, j, k+1))
    
    return neighbors

def backtrack_path(path, start, end):
    # 回溯路径
    i, j, k = end
    
    # 初始化一个列表用于存储路径
    path_list = []
    
    # 从目标点开始回溯
    while (i, j, k) != start:
        path_list.append((i, j, k))
        i, j, k = path[i, j, k]
    
    # 将起始点加入路径列表
    path_list.append(start)
    
    # 将路径反转
    path_list.reverse()
    
    return path_list

# 示例用法
matrix = np.array([
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[0, 1, 0], [0, 1, 0], [0, 0, 0]],
    [[0, 1, 0], [0, 1, 1], [0, 0, 0]]
])

start = (0, 0, 0)
end = (2, 2, 2)

path = find_path(matrix, start, end)

print(path)
