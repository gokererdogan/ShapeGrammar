# -*- coding: utf-8 -*-
'''
Created on May 24, 2013

@author: gerdogan

GraspIt test code
'''

# 
import socket
import numpy as np

host = '0'
port = 4765

conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
conn.connect((host, port))
print 'sending'
conn.send('getDOFVals ALL\n')
print 'receiving'
data = conn.recv(1024)
print data

x = [float(s) for s in data.split('\n') if s is not '']
print x[2:]

conn.close()