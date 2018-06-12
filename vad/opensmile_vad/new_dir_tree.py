#!/usr/bin/env python
import os,sys
dirs=[ r for r,s,f in os.walk(sys.argv[1]) if r != "."]
for i in dirs:
    print(i)
    os.makedirs(os.path.join(sys.argv[2],i))
