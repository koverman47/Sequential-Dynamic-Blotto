#!/usr/bin/env python3

import math
import numpy as np
from time import time
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from scipy.spatial import ConvexHull
from utils import gen_kspace
from node import Node

class Minimax:

    def __init__(self, A, R, V, K, x0, y0, max_depth):
        self.A = A
        self.R = R
        self.V = V
        self.K = K
        self.x0 = x0
        self.y0 = y0
        self.md = max_depth
        self.xspaces = {}
        self.yspaces = {}

    '''
    ' x_cand, y_cand, V
    '''
    def get_payoff(self, xC, yC):
        assert len(xC) == len(yC)
        assert len(xC) == len(self.V)

        win = []
        loss = []
        draw = []
        p = 0.
        for i in range(len(self.V)):
            if xC[i] > yC[i]:
                p += self.V[i]
            elif xC[i] < yC[i]:
                p -= self.V[i]
            if not (xC[i] > yC[i]):
                p -= 0.5 * xC[i]
                p += 0.5 * yC[i]
        return p

    '''
    ' A, K, x
    '''
    def gen_xspace(self, x):
        n = len(x)
        m = len(self.K[0])

        I = np.eye(n)
        U = np.dot(x[0], I)
        for i in range(1, n):
            U = np.concatenate((U, np.dot(x[i], I)), axis=1)
        
        vsave = np.zeros((n, m))
        for i in range(m):
            k = self.K[:, i]
            LS = np.zeros((n, n))
            for j in range(n):
                if k[j] == 1:
                    LS[:, j] = self.A[:, j]

            colsum = np.sum(LS, 0)
            d = np.ones(n) - colsum
            for j in range(n):
                LS[j, j] = d[j]
            
            LSF = LS.flatten('F')
            xv = np.dot(U, LSF)
            vsave[:, i] = xv

        xVsave = np.unique(vsave, axis=1)
        twodee = xVsave[:2, :] 
        indices = np.array(ConvexHull(twodee.T).vertices)
        xV = xVsave.T[indices]

        return xV.T

    '''
    ' xv, R
    '''
    def sample_action(self, xV, n, i):
        r = i % n
        c = math.floor(i / n)
        
        x = r * self.R
        y = c * self.R

        point = Point(x, y)
        poly = []
        for v in xV:
            poly.append((v[0], v[1]))
        polygon = Polygon(poly)
        if polygon.contains(point):
            return [x, y, sum(xV[:, 1]) - x - y]
        return False

    def get_num_nodes(self, node):
        c = 1
        if not node.children:
            return 0
        for child in node.children:
            c += 1 + self.get_num_nodes(child)
        return c

    def recover_solution(self, node):
        n = node
        d = 0
        path = [n]
        while n.children:
            i = -1
            if d % 2 == 0:
                i = n.children.index(max(n.children))
                n = n.children[i]
                path.append(n)
            else:
                i = n.children.index(min(n.children))
                n = n.children[i]
            d += 1
        return path
            

    def run(self):
        t1 = time()
        xV = self.gen_xspace(self.x0)
        yV = self.gen_xspace(self.y0)
        self.xspaces[str(self.x0)] = xV
        self.yspaces[str(self.y0)] = yV
        node = Node(self.x0, self.y0, 0, None)
        value = self.abpruning(node, 0, -1000, 1000, xV, yV)
        t2 = time()
        print("Time", t2 - t1)
        print(value)
        path = self.recover_solution(node)
        print(path)
        print("Number of Nodes : %d" % self.get_num_nodes(node))

    def minimax(self):
        xvert = self.gen_xspace(self.x0)
        XRes = sum(self.x0)
        L = int(XRes / self.R) + 1
        samples = []
        for i in range(L**2):
            x1 = self.sample_action(xvert, L, i)
            if x1 is not False:
                samples.append(x1)

    '''
    ' node, depth, alpha, beta, maximizer
    '''
    def abpruning(self, node, d, a, b, xV, yV, maximizer=True):
        if d == self.md:
            assert d % 2 == 0
            assert not node.children
            p = self.get_payoff(node.x, node.y)
            node.set_payoff(p)
            return node.p
        elif d == 0:
            if str(node.x) in self.xspaces:
                xV = self.xspaces[str(node.x)]
            else:
                xV = self.gen_xspace(node.x)
            value = -1000
            XRes = sum(self.x0)
            L = int(XRes / self.R) + 1
            for i in range(L**2): # number of samples for x-space
                x = self.sample_action(xV, L, i)
                if x is False:
                    continue
                n = Node(x, node.y, d+1, node)
                node.children.append(n)
                # Maximizer?
                assert value is not None
                value = max(value, self.abpruning(n, d+1, a, b, xV, yV))
                if value >= b:
                    break
                a = max(a, value)
            node.set_payoff(value)
            return value
        elif d % 2 == 1:
            if str(node.y) in self.yspaces:
                yV = self.yspaces[str(node.y)]
            else:
                yV = self.gen_xspace(node.y)
            value = 1000
            YRes = sum(self.x0)
            L = int(YRes / self.R) + 1
            for i in range(L**2): # number of samples for y-space
                y = self.sample_action(yV, L, i)
                if y is False:
                    continue
                n = Node(node.x, y, d+1, node)
                node.children.append(n)
                # Minimizer?
                assert value is not None
                result = self.abpruning(n, d+1, a, b, xV, yV)
                value = min(value, self.abpruning(n, d+1, a, b, xV, yV))
                if value <= a:
                    break
                b = min(b, value)
            node.set_payoff(value)
            return node.p
        elif d % 2 == 0:
            if str(node.x) in self.xspaces:
                xV = self.xspaces[str(node.x)]
            else:
                xV = self.gen_xspace(node.x)
            value = -1000
            XRes = sum(self.x0)
            L = int(XRes / self.R) + 1
            for i in range(L**2): # number of samples for x-space
                x = self.sample_action(xV, L, i)
                if x is False:
                    continue
                n = Node(x, node.y, d+1, node)
                node.children.append(n)
                # Maximizer?
                assert value is not None
                value = max(value, self.abpruning(n, d+1, a, b, xV, yV))
                if value >= b:
                    break
                a = max(a, value)
            p = self.get_payoff(node.x, node.y)
            node.set_payoff(p + value)
            return node.p
        print("Returning False")
        print("Depth : %d" % d)


if __name__ == "__main__":
    # Define vars
    md = 2
    A = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    R = 0.1
    V = np.array([1., 1., 1.])

    x0 = np.array([0.3, 0.6, 0.4])
    y0 = np.array([0., 0.2, 0.8])

    # Create K-space vertices
    K = np.array(gen_kspace(A))

    game = Minimax(A, R, V, K, x0, y0, 2 * md)
    #game.minimax()
    game.run()
