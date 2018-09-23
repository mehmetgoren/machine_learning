# -*- coding: utf-8 -*-
"""
object oriented programming in python
"""

class Person:
    height= 180
    
    def run(self, b):
        self.height += b
        return b + 10
    
bitch = Person()

print(bitch.run(12))
print(bitch.height)


def f(x):#parabol
    return x * x;


fptr = f

print(fptr(65))