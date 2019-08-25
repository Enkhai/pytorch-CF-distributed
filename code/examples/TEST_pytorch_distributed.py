# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
#from torch.multiprocessing import Process
import torch.multiprocessing as mp

print('\n', __name__) #debug line


#**READ THIS BEFORE DOING ANYTHING ELSE**: DO NOT initialize
#ANYTHING in cuda beforehand if you intend to work with 
#multiprocessing on cuda. initializing variables in cuda within
#the forked process will cause the program to crash if
#variables have already been initialized outside the processes.
#DO NOT attempt this practice!!!
#Make sure there is nothing on the GPU memory before proceeding
#to initiazize anything in cuda within the processes

#*ADVICE*: upon debugging distributed applications, if the
#program stops unexcepectedly or joins processes without any
#output, run the script using the external console instead of
#the native Spyder console. error messages **CAN BE HIDDEN**.
#similar problems with hidden error messages can infrequently
#occur in other occasions as well

#*NOTICE*: gloo is currently being used as the primary backend
#because of compatibility reasons.
#due to its nature (works on both CPU and GPU) 
#tensors can also be cast to CUDA devices in broadcast and
#all_reduce functions.

#functions to be used
#nccl only supports broadcast, all_reduce,
#reduce and all_gather

#Send/Receive
#send/recv not supported by nccl!
def send_rcv(rank, size):
    """ Distributed function to be implemented later. """
    print(f'Hello from task no. {rank}!')
    tensor = torch.zeros(1)
    if rank == 0:
        print('I increase the tensor value by 1\
 and send it to process 1!')
        tensor += 1
        dist.send(tensor=tensor, dst=1)
    else:
        print('And I receive it!')
        dist.recv(tensor=tensor, src=0)
    print(f'My tensor has this data: {tensor[0]}')
    #pass

#isend/irecv (asynchronous)
#send/recv not supported by nccl!
def isend_ircv(rank, size):
    """ Distributed function to be implemented later. """
    print(f'Hello from task no. {rank}!')
    tensor = torch.zeros(1)
    req = None
    if rank == 0:
        print('I increase the tensor value by 1\
 and send it to process 1!')
        tensor += 1
        req = dist.isend(tensor=tensor, dst=1)
        print('Task 0 started sending')
    else:
        print('And I receive it!')
        req = dist.irecv(tensor=tensor, src=0)
        print('Task 1 started receiving')
    #wait to receive/send data
    req.wait()
    print(f'Tasks synchonized!')
    print(f'My tensor has this data: {tensor[0]}')
    #pass

#all_reduce    
def all_reduce(rank, size):
    print('Hello from task ', rank, '!')
    #create a new group with the following
    #processes
    #Notice: all processes belonging in the
    #distributed job MUST be entered, even if
    #they are not to be used.
    #Groups should also be created in the same
    #order in all processes
    group = dist.new_group([0,1])
    #make a simple tensor
    tensor = torch.ones(1)#.cuda()
    print('Before:')
    print('Rank ', rank, ' has data ', tensor[0])
    #everybody in the group do an all reduce
    #operation summing up the tensors received
    #by everyone else
    dist.all_reduce(tensor=tensor, op=dist.reduce_op.SUM, group=group)
    print('After:')
    print('Rank ', rank, ' has data ', tensor[0])
    
    
#more functions
    
#supported by nccl
def broadcast(rank, size):
    pass

def reduce(rank, size):
    pass

def all_gather(rank, size):
    pass


#not supported by nccl
def gather(rank, size):
    pass

def scatter(rank, size):
    pass

def barrier(rank, size):
    pass

    
#initialize processes
#declare rank, size of group, function to be run
#and a backend.

#master address, port and world size can be declared outside 
#of the processes as well (global)
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'
os.environ['WORLD_SIZE'] = str(2)

#backend in gloo because nccl times out in Kepler 3.0 even when
#built from source...
def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    #os.environ['MASTER_ADDR'] = '127.0.0.1'
    #os.environ['MASTER_PORT'] = '29500'
    #os.environ['WORLD_SIZE'] = str(size)
    os.environ['RANK'] = str(rank)
    dist.init_process_group(backend, world_size=size, rank=rank)
    fn(rank, size)
    

#spawn and forkserver methods don't work inside functions...
#def do():
#it is a best practice to use the if __name__=='main' section.
#start methods like spawn and forkserver will produce a copy
#of the process's parent resulting in continuous process
#creation and a runtime error. using the if section ensures
#that no process creation will occur any further, since the
#__name__ variable's value will not be 'main' in the created
#process but a script name instead

#NEEDS TESTING TO BE USED INSTEAD OF FILE SAVING!!!
print('task calls', os.environ['TASK_CALLS'])

import sys
sys.path.append('../')
import pickle
fr = open('shared.pkl', 'rb')
try:
    iteration = pickle.load(fr)
except:
    pass

try:
    if iteration == 0:
        fw = open('shared.pkl', 'wb')
        #pickle.dump(1, fw)
        #if __name__ == "__main__":
        #if __name__ == "examples.pytorch_distributed":
            #__name__ will refer to the function and not the script
            #if set within a function.
            #the functions called below will not belong to the
            #main function therefore initialization will fail with 
            #spawn and forkserver methods.
            #also, despite expected outcome, fork method (default)
            #will work with cuda tensors using distributed operations!!
            #if there is no specific worry about sharing tensors
            #otherwise with multiprocessing, it seems obvious that
            #any start method will produce similar results with the
            #distributed package operations.
            #furthermore, it is a best practice to set the start
            #method within the if __name__=='main' section.
            #the spawn method can also be set only *once* inside
            #a **main** script!!!
            #mp.set_start_method('fork') #default
            #or
            #mp.set_start_method('spawn')
            #or
            #mp.set_start_method('forkserver')
        size = 2
        processes = []
        for rank in range(size):
            print('Initializing task ', rank)
            p = mp.Process(target=init_processes, args=(rank, size, all_reduce))
            print('Run process of task ', rank)
            p.start()
            processes.append(p)
        
        print('\nActive processes before joining: ', mp.active_children())
        print('Joining ', len(mp.active_children()) ,' processes')
        for p in processes:
            p.join()
                
        print('Active processes after joining: ', mp.active_children())
except: pass