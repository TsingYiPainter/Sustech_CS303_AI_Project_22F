import copy
import getopt, sys
import math
import random
import time
import multiprocessing as mp
import numpy as np


class CARP:
    def __init__(self, vertices, depot, required_Edge, non_required, vehicles, capacity, total_cost, map):
        self.vertices = vertices
        self.depot = depot
        self.req_Edge = required_Edge
        self.n_req = non_required
        self.vehicles = vehicles
        self.capacity = capacity
        self.total_c = total_cost
        self.map = {(map[i][0], map[i][1]): [map[i][2], map[i][3]] for i in range(len(map))}
        self.distanceMap = self.Floyd(map, vertices)
        self.oriFree = {}
        for items in self.map.items():
            if items[1][1] != 0:
                self.oriFree[items[0]] = items[1]
                self.oriFree[(items[0][1], items[0][0])] = items[1]

    def Floyd(self, map, num):
        max = 1_000_000
        Edge_Count = self.req_Edge + self.n_req
        new_map = np.array([[max] * num] * num)
        for i in range(len(new_map)):
            new_map[i][i] = 0
        for i in range(Edge_Count):
            new_map[map[i][0] - 1][map[i][1] - 1] = map[i][2]
            new_map[map[i][1] - 1][map[i][0] - 1] = map[i][2]
        for k in range(num):
            for i in range(num):
                for j in range(num):
                    new_map[i][j] = min(new_map[i][j], new_map[i][k] + new_map[k][j])
        # print(new_map)
        return new_map

    def cal_cost(self, route):
        cost = 0
        load = 0
        cur_End = self.depot
        for i in range(len(route)):
            cur_Path = route[i]
            cur_load = self.oriFree[cur_Path][1]
            cur_cost = self.oriFree[cur_Path][0]
            if load + cur_load <= self.capacity:
                cost += self.distanceMap[cur_End - 1][cur_Path[0] - 1] + cur_cost
                load += cur_load
                cur_End = cur_Path[1]
            else:
                load = 0
                load += cur_load
                cost += self.distanceMap[cur_End - 1][self.depot - 1] + self.distanceMap[self.depot - 1][
                    cur_Path[0] - 1]
                cost += cur_cost
                cur_End = cur_Path[1]
        cost += self.distanceMap[cur_End - 1][self.depot - 1]
        return cost


class Sol:
    def __init__(self, carp):
        self.population = []
        self.remain = [[np.inf, 0]]
        self.population_size = 100
        self.problem = carp

    def init_population(self, carp, size):

        capacity = carp.capacity
        for i in range(size):
            load, cost = 0, 0
            cur_End = carp.depot
            routes = []
            route = []
            task = []
            remian = []
            free = copy.deepcopy(carp.oriFree)

            while free != {} or route != []:
                min_D = 1_000_000
                for _ in free.items():
                    if load + _[1][1] <= capacity:
                        cur_D = carp.distanceMap[cur_End - 1][_[0][0] - 1]
                        if cur_D < min_D:
                            task = [_[0]]
                            min_D = cur_D
                        elif cur_D == min_D:
                            task.append(_[0])
                if task == []:
                    # res.append((cur_End, carp.depot))
                    # res.append((0, 0))
                    routes.append(route)
                    route = []
                    cost += carp.distanceMap[cur_End - 1][carp.depot - 1]
                    remian.append([cost, capacity - load])
                    load = 0
                    cur_End = carp.depot
                else:
                    randonInt = random.randint(0, len(task) - 1)
                    roundRes = task[randonInt]
                    task = []
                    route.append(roundRes)
                    load += free[roundRes][1]
                    cost += free[roundRes][0] + carp.distanceMap[cur_End - 1][roundRes[0] - 1]
                    del free[roundRes]
                    del free[(roundRes[1], roundRes[0])]
                    cur_End = roundRes[1]
            # self.population.append((res, cost + carp.distanceMap[cur_End - 1][carp.depot - 1]))
            if remian[len(remian) - 1][0] < self.remain[len(self.remain) - 1][0]:
                self.population = routes
                self.remain = remian
        self.two_opt1(self.population, self.remain)
        # self.flip(self.population, self.remain)
        # print_res(self.population, self.remain)

    def two_opt1(self, population, remain):
        for i in range(len(population)):
            length = len(population[i])
            good_res = population[i]
            good_remain = 0
            for index1 in range(length - 1):
                for index2 in range(index1 + 1, length):
                    tmp_route = copy.deepcopy(population[i])
                    if index1 == 0:
                        beginPoint = self.problem.depot
                    else:
                        beginPoint = tmp_route[index1 - 1][1]

                    endPoint = tmp_route[index2 - 1][1]
                    dis1 = self.problem.distanceMap[beginPoint - 1][tmp_route[index1][0] - 1]
                    dis2 = self.problem.distanceMap[endPoint - 1][tmp_route[index2][0] - 1]

                    self.my_resvere(tmp_route, index1, index2)

                    if index1 != 0:
                        beginPoint = tmp_route[index1 - 1][1]
                    endPoint = tmp_route[index2 - 1][1]
                    dis1_ = self.problem.distanceMap[beginPoint - 1][tmp_route[index1][0] - 1]
                    dis2_ = self.problem.distanceMap[endPoint - 1][tmp_route[index2][0] - 1]
                    tmp_diff = (dis1 + dis2 - dis1_ - dis2_)
                    if tmp_diff > 0:
                        good_res = tmp_route
                        good_remain = tmp_diff
            population[i] = good_res
            for k in range(i, len(remain)):
                remain[k][0] = remain[k][0] - good_remain

    def my_resvere(self, route, a, b):
        route[a:b] = reversed(route[a:b])
        route[a:b] = [(route[i][1], route[i][0]) for i in range(a, b)]

    def flip(self, population, remain):
        for i in range(len(population)):
            length = len(population[i])
            for k in range(length):
                task = population[i][k]
                if k == 0:
                    beginPoint = self.problem.depot
                else:
                    beginPoint = population[i][k - 1][1]
                if k == length - 1:
                    endPoint = self.problem.depot
                else:
                    endPoint = population[i][k + 1][0]
                dis1 = self.problem.distanceMap[beginPoint - 1][task[0] - 1]
                dis2 = self.problem.distanceMap[task[1] - 1][endPoint - 1]
                dis1_ = self.problem.distanceMap[beginPoint - 1][task[1] - 1]
                dis2_ = self.problem.distanceMap[task[0] - 1][endPoint - 1]

                diff = dis1 + dis2 - dis1_ - dis2_
                if diff > 0:
                    for _ in range(i, len(remain)):
                        remain[_][0] -= diff
                    population[i][k] = (task[1], task[0])

    def self_single_insertion(self, population, remain):

        while True:
            select = random.randint(0, len(population) - 1)
            route = population[select]
            if len(route) != 1:
                break

        select_task = random.randint(0, len(route) - 1)
        task = route[select_task]
        if select_task == 0:
            beginPoint = self.problem.depot
        else:
            beginPoint = route[select_task - 1][1]
        if select_task == len(route) - 1:
            endPoint = self.problem.depot
        else:
            endPoint = route[select_task + 1][0]
        dis1 = self.problem.distanceMap[beginPoint - 1][task[0] - 1]
        dis2 = self.problem.distanceMap[task[1] - 1][endPoint - 1]
        dis3 = self.problem.distanceMap[beginPoint - 1][endPoint - 1]
        route.remove(task)

        idx_task = random.randint(0, len(route))
        if idx_task == 0:
            beginPoint = self.problem.depot
        else:
            beginPoint = route[idx_task - 1][1]
        if idx_task == len(route):
            endPoint = self.problem.depot
        else:
            endPoint = route[idx_task][0]
        route.insert(idx_task, task)
        dis1_ = self.problem.distanceMap[beginPoint - 1][task[0] - 1]
        dis2_ = self.problem.distanceMap[task[1] - 1][endPoint - 1]
        dis3_ = self.problem.distanceMap[beginPoint - 1][endPoint - 1]

        diff = dis1 + dis2 - dis3 - dis1_ - dis2_ + dis3_
        for i in range(select, len(remain)):
            remain[i][0] -= diff
        return diff

    def cross_single_insertion(self, population, remain):
        while True:
            select = random.randint(0, len(population) - 1)
            route = population[select]
            if len(route) != 1:
                break

        select_task = random.randint(0, len(route) - 1)
        task = route[select_task]
        if select_task == 0:
            beginPoint = self.problem.depot
        else:
            beginPoint = route[select_task - 1][1]
        if select_task == len(route) - 1:
            endPoint = self.problem.depot
        else:
            endPoint = route[select_task + 1][0]
        dis1 = self.problem.distanceMap[beginPoint - 1][task[0] - 1]
        dis2 = self.problem.distanceMap[task[1] - 1][endPoint - 1]
        dis3 = self.problem.distanceMap[beginPoint - 1][endPoint - 1]
        required = self.problem.oriFree[task][1]
        task_cost = self.problem.oriFree[task][0]
        # remain[select][0] = remain[select][0] - dis1 - dis2 + dis3

        idx_route = 0
        flag = False
        for select_route in range(len(remain)):
            if remain[select_route][1] > required and select_route != select:
                idx_route = select_route
                flag = True
                break
        if not flag:
            return 0
        route.remove(task)
        remain[select][1] += required

        for i in range(select, len(remain)):
            remain[i][0] -= (task_cost + dis1 + dis2 - dis3)

        remain[idx_route][1] -= required
        route = population[idx_route]

        idx_task = random.randint(0, len(route))
        if idx_task == 0:
            beginPoint = self.problem.depot
        else:
            beginPoint = route[idx_task - 1][1]
        if idx_task == len(route):
            endPoint = self.problem.depot
        else:
            endPoint = route[idx_task][0]
        route.insert(idx_task, task)
        dis1_ = self.problem.distanceMap[beginPoint - 1][task[0] - 1]
        dis2_ = self.problem.distanceMap[task[1] - 1][endPoint - 1]
        dis3_ = self.problem.distanceMap[beginPoint - 1][endPoint - 1]
        for i in range(idx_route, len(remain)):
            remain[i][0] += (task_cost + dis1_ + dis2_ - dis3_)

        diff = dis1 + dis2 - dis3 - dis1_ - dis2_ + dis3_
        return diff

    def cross_double_insertion(self, population, remain):
        while True:
            select = random.randint(0, len(population) - 1)
            route = population[select]
            if len(route) != 1 and len(route) != 2:
                break

        select_task = random.randint(0, len(route) - 2)
        task = route[select_task:select_task + 2]
        if select_task == 0:
            beginPoint = self.problem.depot
        else:
            beginPoint = route[select_task - 1][1]
        if select_task == len(route) - 2:
            endPoint = self.problem.depot
        else:
            endPoint = route[select_task + 2][0]
        dis1 = self.problem.distanceMap[beginPoint - 1][task[0][0] - 1]
        dis2 = self.problem.distanceMap[task[1][1] - 1][endPoint - 1]
        dis3 = self.problem.distanceMap[beginPoint - 1][endPoint - 1]
        required = self.problem.oriFree[task[0]][1] + self.problem.oriFree[task[1]][1]
        task_cost = self.problem.oriFree[task[0]][0] + self.problem.oriFree[task[1]][0]

        idx_route = 0
        flag = False
        for select_route in range(len(remain)):
            if remain[select_route][1] > required and select_route != select:
                idx_route = select_route
                flag = True
                break
        if not flag:
            return 0

        route.remove(task[0])
        route.remove(task[1])
        remain[select][1] += required

        for i in range(select, len(remain)):
            remain[i][0] -= (task_cost + dis1 + dis2 - dis3)

        remain[idx_route][1] -= required
        route = population[idx_route]

        idx_task = random.randint(0, len(route))
        if idx_task == 0:
            beginPoint = self.problem.depot
        else:
            beginPoint = route[idx_task - 1][1]
        if idx_task == len(route):
            endPoint = self.problem.depot
        else:
            endPoint = route[idx_task][0]
        route.insert(idx_task, task[0])
        route.insert(idx_task + 1, task[1])
        dis1_ = self.problem.distanceMap[beginPoint - 1][task[0][0] - 1]
        dis2_ = self.problem.distanceMap[task[1][1] - 1][endPoint - 1]
        dis3_ = self.problem.distanceMap[beginPoint - 1][endPoint - 1]
        for i in range(idx_route, len(remain)):
            remain[i][0] += (task_cost + dis1_ + dis2_ - dis3_)

        diff = dis1 + dis2 - dis3 - dis1_ - dis2_ + dis3_
        return diff

    def swap(self, routes, remain):
        select1, select2 = 0, 0
        while (select1 == select2):
            select1 = random.randint(0, len(routes) - 1)
            select2 = random.randint(0, len(routes) - 1)
        route1 = routes[select1]
        route2 = routes[select2]

        counter = 0
        while True:
            counter += 1
            select_task1 = random.randint(0, len(route1) - 1)
            select_task2 = random.randint(0, len(route2) - 1)
            task1 = route1[select_task1]
            task2 = route2[select_task2]

            task1_cost, task1_load = self.problem.oriFree[task1]
            task2_cost, task2_load = self.problem.oriFree[task2]
            if counter == len(route1) + len(route2):
                return 0
            if not (remain[select1][1] + task1_load - task2_load >= 0 and remain[select2][
                1] + task2_load - task1_load >= 0):
                continue

            if select_task1 == 0:
                beginPoint1 = self.problem.depot
            else:
                beginPoint1 = route1[select_task1 - 1][1]
            if select_task1 == len(route1) - 1:
                endPoint1 = self.problem.depot
            else:
                endPoint1 = route1[select_task1 + 1][0]

            if select_task2 == 0:
                beginPoint2 = self.problem.depot
            else:
                beginPoint2 = route2[select_task2 - 1][1]
            if select_task2 == len(route2) - 1:
                endPoint2 = self.problem.depot
            else:
                endPoint2 = route2[select_task2 + 1][0]

            dis1 = -self.problem.distanceMap[beginPoint1 - 1][task1[0] - 1] + self.problem.distanceMap[beginPoint1 - 1][
                task2[0] - 1]
            dis2 = -self.problem.distanceMap[task1[1] - 1][endPoint1 - 1] + self.problem.distanceMap[task2[1] - 1][
                endPoint1 - 1]

            dis3 = -self.problem.distanceMap[beginPoint2 - 1][task2[0] - 1] + self.problem.distanceMap[beginPoint2 - 1][
                task1[0] - 1]
            dis4 = -self.problem.distanceMap[task2[1] - 1][endPoint2 - 1] + self.problem.distanceMap[task1[1] - 1][
                endPoint2 - 1]

            diff1 = dis1 + dis2 - task1_cost + task2_cost
            diff2 = dis3 + dis4 - task2_cost + task1_cost
            for i in range(select1, len(remain)):
                remain[i][0] += diff1
            for i in range(select2, len(remain)):
                remain[i][0] += diff2
            remain[select1][1] += (task1_load - task2_load)
            remain[select2][1] += (task2_load - task1_load)
            route1.remove(task1)
            route2.remove(task2)
            route1.insert(select_task1, task2)
            route2.insert(select_task2, task1)
            return diff1 + diff2

    def re_combine(self, routes, remain):
        select1 = random.randint(0, len(routes) - 1)
        select2 = random.randint(0, len(routes) - 1)
        while select1 == select2:
            select1 = random.randint(0, len(routes) - 1)
            select2 = random.randint(0, len(routes) - 1)
        route1 = routes[select1]
        route2 = routes[select2]
        cost1 = self.cal_oneRoute(route1)
        cost2 = self.cal_oneRoute(route2)
        mid1 = random.randint(0, len(route1) - 1)
        mid2 = random.randint(0, len(route2) - 1)
        count = 0
        while True:
            count += 1
            new_route1 = route1[:mid1] + route2[mid2:]
            new_route2 = route2[:mid2] + route1[mid1:]
            sum1, sum2 = 0, 0
            for i in range(0, len(new_route1)):
                sum1 += self.problem.oriFree[new_route1[i]][1]
            for i in range(0, len(new_route2)):
                sum2 += self.problem.oriFree[new_route2[i]][1]
            if (sum1 <= self.problem.capacity and sum2 <= self.problem.capacity) or count > len(route1) + len(route2):
                break
        if count > len(new_route1) + len(new_route2):
            return 0

        new_cost1, new_cost2 = self.cal_oneRoute(new_route1), self.cal_oneRoute(new_route2)
        diff1 = cost1 - new_cost1
        diff2 = cost2 - new_cost2
        routes[select1] = new_route1
        routes[select2] = new_route2
        for i in range(select1, len(routes)):
            remain[i][0] -= diff1
        for i in range(select2, len(routes)):
            remain[i][0] -= diff2
        remain[select1][1] = self.problem.capacity - sum1
        remain[select2][1] = self.problem.capacity - sum2
        return diff1 + diff2
        # cost += self.problem.distanceMap[end - 1][population[i][k][0] - 1] + \
        #         self.problem.oriFree[population[i][k]][0]
        # end = population[i][k][1]
        #

    def merge_split(self, routes, remain):
        select1 = random.randint(0, len(routes) - 1)
        select2 = random.randint(select1, len(routes) - 1)
        while(select1==select2):
            select1 = random.randint(0, len(routes) - 1)
            select2 = random.randint(select1, len(routes) - 1)
        tmp_Free = {}
        all_cost = 0
        if select1==0:
            all_cost=remain[select2][0]
        else:
            all_cost=remain[select2][0]-remain[select1-1][0]
        for i in range(select1, select2+1):
            route = routes[i]
            for j in range(len(route)):
                tmp_Free[route[j]] = self.problem.oriFree[route[j]]
                tmp_Free[(route[j][1], route[j][0])] = self.problem.oriFree[route[j]]
        for i in range(select1,select2+1):
            route=routes[select1]
            routes.remove(route)
            remain.remove(remain[select1])

        for i in range(select1, len(routes)):
            remain[i][0] -= all_cost

        load, cost = 0, 0
        cur_End = self.problem.depot
        capacity = self.problem.capacity
        depot = self.problem.depot
        cur_routes = []
        route = []
        task = []
        cur_remain = []
        free = tmp_Free
        randonInt = random.randint(0, 4)
        while free != {} or route != []:
            min_D = 1_000_000
            for _ in free.items():
                if load + _[1][1] <= capacity:
                    cur_D = self.problem.distanceMap[cur_End - 1][_[0][0] - 1]
                    if cur_D < min_D:
                        task = [_[0]]
                        min_D = cur_D
                    elif cur_D == min_D:
                        task.append(_[0])
            if task == []:
                # res.append((cur_End, carp.depot))
                # res.append((0, 0))
                cur_routes.append(route)
                route = []
                cost += self.problem.distanceMap[cur_End - 1][depot - 1]
                cur_remain.append([cost, capacity - load])
                load = 0
                cur_End = depot
            else:
                if randonInt == 0:
                    roundRes = task[random.randint(0, len(task) - 1)]
                elif randonInt == 1:
                    task = sorted(task, key=lambda k: self.problem.distanceMap[k[0] - 1][depot - 1])
                    roundRes = task[0]
                elif randonInt == 2:
                    task = sorted(task, key=lambda k: self.problem.distanceMap[k[0] - 1][depot - 1])
                    roundRes = task[len(task) - 1]
                elif randonInt == 3:
                    task = sorted(task, key=lambda k: free[k][0] / free[k][1])
                    roundRes = task[0]
                else:
                    task = sorted(task, key=lambda k: free[k][0] / free[k][1])
                    roundRes = task[len(task) - 1]
                task = []
                route.append(roundRes)
                load += free[roundRes][1]
                cost += free[roundRes][0] + self.problem.distanceMap[cur_End - 1][roundRes[0] - 1]
                del free[roundRes]
                del free[(roundRes[1], roundRes[0])]
                cur_End = roundRes[1]
        # self.population.append((res, cost + carp.distanceMap[cur_End - 1][carp.depot - 1]))

        copy_select1 = select1
        for i in range(len(cur_routes)):
            routes.insert(select1, cur_routes[i])
            remain.insert(select1, cur_remain[i])
            if select1 != 0:
                remain[select1][0] += remain[select1 - 1][0]
            select1 += 1
        if copy_select1 == 0:
            all_cost = remain[select1 - 1][0]
        else:
            all_cost = remain[select1 - 1][0] - remain[copy_select1 - 1][0]
        for i in range(select2+1, len(routes)):
            remain[i][0] += all_cost


    def cal_oneRoute(self, route):
        cost = 0
        end = self.problem.depot
        for i in range(len(route)):
            cost += self.problem.distanceMap[end - 1][route[i][0] - 1] + \
                    self.problem.oriFree[route[i]][0]
            end = route[i][1]
        cost += self.problem.distanceMap[end - 1][self.problem.depot - 1]
        return cost

    def simulated_annealing(self, weight, routes, remain):
        best_routes, new_routes, old_routes = copy.deepcopy(routes), copy.deepcopy(routes), copy.deepcopy(routes)
        best_remain, new_remain, old_remain = copy.deepcopy(remain), copy.deepcopy(remain), copy.deepcopy(remain)
        T = 10000
        cool_rate = 0.95
        best_len=len(best_remain)
        repeat = 0
        index = 0
        index1 = 0
        index2 = 0
        # print(time.time())
        # print(start)
        # print(runtime - 2)
        while time.time() - start < runtime - 2:
            index += 1
            if time.time() - start >= runtime - 2:
                # print("out!!!!!")
                return best_routes, best_remain
            randInt = random.randint(0, 24)
            copy_routes = copy.deepcopy(new_routes)
            copy_remain = copy.deepcopy(new_remain)
            if randInt <= weight[0]:
                self.flip(copy_routes, copy_remain)
            elif randInt <= weight[1]:
                self.self_single_insertion(copy_routes, copy_remain)
            elif randInt <= weight[2]:
                self.cross_single_insertion(copy_routes, copy_remain)
            elif randInt <= weight[3]:
                self.two_opt1(copy_routes, copy_remain)
            elif randInt <= weight[4]:
                self.re_combine(copy_routes, copy_remain)
            elif randInt <= weight[5]:
                self.swap(copy_routes, copy_remain)
            elif randInt <= weight[6]:
                self.cross_double_insertion(copy_routes, copy_remain)
            else:
                self.merge_split(copy_routes,copy_remain)
            diff = new_remain[len(new_remain) - 1][0] - copy_remain[len(copy_remain) - 1][0]
            flag = False
            if diff < 0:
                flag = math.exp(diff * random.randint(40, 80) / T) > random.random()
                index1 += 1
                if flag:
                    index2 += 1
            if diff >= 0 or flag:
                new_routes = copy_routes
                new_remain = copy_remain

                repeat = 0
            if best_remain[best_len - 1][0] > copy_remain[len(copy_remain) - 1][0]:
                best_routes = copy_routes
                best_remain = copy_remain
                best_len=len(copy_remain)
            if diff == 0:
                # print(randInt)
                repeat += 1
            if copy_remain[len(copy_remain) - 1][0] >= 1.2 * best_remain[best_len - 1][0]:
                T = 10000
                new_routes = best_routes
                new_remain = best_remain

            if repeat > 50:
                T = 10000
                repeat = 0

            # print("index={0},diff={1},T={2}".format(index, diff, T))
            # print("current score:")
            # print(new_remain[len(new_remain) - 1][0])
            # print("best score")
            # print(best_remain[best_len - 1][0])
                # shijian.append(time.time()-start)
                # fenshu.append(new_remain[len(new_remain)-1][0])
                # wendu.append(T)



            # print(index1)
            # print(index2)

            T = T * cool_rate
        # print(shijian)
        # print(fenshu)
        # print(wendu)
        return best_routes, best_remain

    def print_res(self, population, remain):
        str1 = "s 0"
        str2 = ""
        cost = 0
        end = 1
        for i in range(len(population)):
            for k in range(len(population[i])):
                str2 += (",({}".format(population[i][k][0]) + ",{})".format(population[i][k][1]))
                cost += self.problem.distanceMap[end - 1][population[i][k][0] - 1] + \
                        self.problem.oriFree[population[i][k]][0]
                end = population[i][k][1]
            if i != len(population) - 1:
                str2 += ",0,0"
                cost += self.problem.distanceMap[end - 1][0]
                end = 1
        cost += self.problem.distanceMap[end - 1][0]
        Str = str1 + str2 + ",0"
        print(Str)
        print("q {}".format(remain[len(remain) - 1][0]))
        #print(cost)


def my_main(step, weight):
    argv = sys.argv
    opts, args = getopt.getopt(argv[2:], "t:s:", [])
    path = sys.argv[1]
    myTime = opts[0][1]
    mySeed = int(opts[1][1])
    random.seed(mySeed + step)
    global start
    start = time.time()
    global runtime
    runtime = int(myTime)
    f = open(path, mode='r', encoding='utf-8')

    sentimentlist = []
    for line in f:
        s = line.strip().split('\t')
        if s[0] == 'END':
            break
        # print(s)
        # print(type(s))
        sentimentlist.append(s)

    f.close()

    smapleName = sentimentlist[0][0].split()[2]
    vertices = int(sentimentlist[1][0].split()[2])
    depot = int(sentimentlist[2][0].split()[2])
    required_Edge = int(sentimentlist[3][0].split()[3])
    non_required = int(sentimentlist[4][0].split()[3])
    Edge_C = required_Edge + non_required
    vehicles = int(sentimentlist[5][0].split()[2])
    capacity = int(sentimentlist[6][0].split()[2])
    total_cost = int(sentimentlist[7][0].split()[6])
    map = sentimentlist[9:]
    map = [map[i][0].split() for i in range(Edge_C)]

    map = [[int(map[i][j]) for j in range(4)] for i in range(Edge_C)]
    carp = CARP(vertices, depot, required_Edge, non_required, vehicles, capacity, total_cost, map)
    my_Sol = Sol(carp)
    my_Sol.init_population(carp, 10000)
    best_routes, best_remain = my_Sol.simulated_annealing(weight, my_Sol.population, my_Sol.remain)
    # my_Sol.print_res(best_routes, best_remain)
    return best_routes, best_remain
    # for i in range(1):
    #     my_GA.select(i)
    #     # print("random: {}".format(random.randint(1,100)))
    #     print_res(my_GA.population[0], carp)


class Pro(mp.Process):
    def __init__(self, arg, res):
        mp.Process.__init__(self, target=self.start)
        self.arg = arg
        self.res = res
        self.exit = mp.Event()

    def run(self):
        while True:
            i, weight = self.arg.get()
            r, c = my_main(i, weight)
            cost = [c[len(c) - 1][0]]
            self.res.put((r, cost))
            # print(time.time() - start_time)


def print_res(population, remain):
    str1 = "s 0"
    str2 = ""
    cost = 0
    end = 1
    for i in range(len(population)):
        for k in range(len(population[i])):
            str2 += (",({}".format(population[i][k][0]) + ",{})".format(population[i][k][1]))
        if i != len(population) - 1:
            str2 += ",0,0"
    Str = str1 + str2 + ",0"
    print(Str)
    print("q {}".format(remain[0]))


if __name__ == '__main__':
    argv = sys.argv
    # main(sys.argv)
    pro = []

    # filp,self_single,self_cross,2-opt,recombine,swap
    weight = [[1, 4, 7, 12, 18, 20, 23], [1, 4, 7, 12, 18, 20, 23], [1, 4, 7, 12, 18, 20, 23], [1, 4, 7, 12, 18, 20, 23],
              [4, 8, 12, 14, 16, 20, 23], [4, 8, 12, 14, 16, 20, 23], [1, 3, 5, 11, 18, 20, 23], [1, 3, 5, 11, 18, 20, 23]]
    for i in range(8):
        pro.append(Pro(mp.Queue(), mp.Queue()))
        pro[i].start()
        pro[i].arg.put([i, weight[i]])
    result = []
    for i in range(8):
        result.append(pro[i].res.get())
    list = sorted(result, key=lambda res: res[1])
    ans = list[0][0]
    cost = list[0][1]
    print_res(ans, cost)

    for p in pro:
        p.terminate()
