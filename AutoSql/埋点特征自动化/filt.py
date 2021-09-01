#!/usr/bin/env python
import sys
startsig = '$AppStart'
endsig = '$AppEnd'
lastkey = None
for i,line in enumerate(sys.stdin): 
    line = line.strip().split("\t")
    key = line[0]
    eventid = line[1]
    timestamp = line[2]
    if key!=lastkey:
        index_tmp = 0
        group = []
        lastkey = None
        set_id = 0    
        endid = 0    
        startid = None
    group.append(line) 
    if eventid==endsig:
        endid = index_tmp
    elif endid is not None and eventid==startsig:
        startid = index_tmp
        for j in range(endid,startid+1):
            print('\t'.join(group[j]+[str(set_id)]))
        endid = None
        startid = None
        set_id+=1
    index_tmp += 1
    lastkey = key
