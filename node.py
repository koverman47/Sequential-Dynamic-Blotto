#!/usr/bin/env python3

class Node:

    def __init__(self, x, y, d, parent):
        self.x = x
        self.y = y
        self.d = d
        self.p = 0.
        self.children = []
        self.parent = parent

    def set_payoff(self, p):
        self.p = p

    def __lt__(self, other):
        return self.p < other.p

    def __eq__(self, other):
        return self.p == other.p

    def __repr__(self):
        return(str([self.x, self.y]))
