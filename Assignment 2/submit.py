import copy
import time
#do not modify the function names
#You are given L and M as input
#Each of your functions should return the minimum possible L value alongside the marker positions
#Or return -1,[] if no solution exists for the given L

#Your backtracking function implementation

minimum=100
ansList=[]
nodeCount=0

'''This method is used for constructing
   the csp graph for given M and L
   Graph will contain M nodes and domain
   will be from 0 to L'''

def constructCSP(L,M):
    graph=dict()
    for i in xrange(M):
        temp=set()
        for j in xrange(L+1):
            temp.add(j)
        graph[i]=temp
    return graph

'''This method is used for making the graph
    arc consistent.If a node has a single value
    in the domain it will remove that val+x 
    from all other nodes'''

def makeArcConsistent(graph,L,M):
    for i in xrange(M-1):
        if len(graph[i]) is 1:
            element = next(iter(graph[i]))
            for j in xrange(i):
                for x in graph[j]:
                    if element-x in graph[j]:
                        graph[j].remove(element-x)
    return graph

'''A helper method for counting 
    number of nodes'''

def incrementNodeCount():
    global nodeCount
    nodeCount+=1

'''Backtracking method called after calling
    arc consistency on smaller graph'''

def csp_backtrack(graph,M,L,current,ans,distance):
    global minimum,ansList
    ''' base case'''
    if(current==M-1):
	value=ans[len(ans)-1]
        incrementNodeCount()
	if(value<minimum):
		minimum=value
		ansList=copy.deepcopy(ans)
    else:
        domain=graph[current+1]
        value=ans[len(ans)-1]
        for x in domain:
	    if(x>value):
                incrementNodeCount()
                check=True
                for a in ans:
                    if x-a in distance:
                        check=False
                if check:
                    ''' optimization : maintaining a distance set'''
                    for a in ans:
                        distance.add(x-a)
                    ans.append(x)
                    ''' calling backtrack again on arc-consistent graph'''
                    csp_backtrack(graph,M,L,current+1,ans,distance)
                    ans.remove(x)
                    for a in ans:
                        distance.remove(x-a)

''' Our backtracking function csp graph has m nodes
    and each node has domain of 0-L.
    Our constraints are that for selected length
    distance of L to all prev selected L be distinct.
    we will backtrack if no such length found
    '''

def backtrack(graph,M,L,current,ans):
    global minimum,ansList
    '''base case'''
    if(current==M-1):
	value=ans[len(ans)-1]
        incrementNodeCount()
	if(value<minimum):
		minimum=value
		ansList=copy.deepcopy(ans)
    else:
        domain=graph[current+1]
        value=ans[len(ans)-1]
        for x in domain:
	    if(x>value):
                incrementNodeCount()
                check=True
		distance=[]
                for a in ans:
                    for b in ans:
                        if a is not b:
				distance.append(abs(a-b))

		for a in ans:
			if x-a in distance:
				check=False;
                if check:
                    ans.append(x)
                    ''' calling backtrack again'''
                    backtrack(graph,M,L,current+1,ans)
                    ans.remove(x)

''' Calling our forward check function
    Once i select a length for a node
    I will remove the distances in our ans and this 
    node from all connected nodes.
    we can see that nodes discovered in this method
    are lesser than backtract and time is also reduced.
    '''

def forward(graph,M,L,current,ans,distance):
    global minimum,ansList,nodeCount
    ''' base case '''
    if(current==M-1):
        value=ans[len(ans)-1]
        incrementNodeCount()
        if(value<minimum):
            minimum=value
            ansList=copy.deepcopy(ans)
    else:
        value=ans[len(ans)-1]
        ''' removing the domain from connected nodes '''
        for d in distance:
            for a in ans:
                if a+d in graph[current+1]:
                    graph[current+1].remove(a+d)

        domain=graph[current+1]

        for x in domain:
            if(x>value and x<minimum):
                incrementNodeCount()
                for a in ans:
                    distance.add(x-a)
                ans.append(x)
                ''' calling forward check on reduced domain graph'''
                forward(graph,M,L,current+1,ans,distance)
                ans.remove(x)
                for a in ans:
                    if x-a in distance:
                        distance.remove(x-a)

        ''' need to add back for backtracking cases'''
        for d in distance:
            for a in ans:
                if a+d <=L:
                    graph[current+1].add(a+d)


def BT(L, M):
    "*** YOUR CODE HERE ***"
    print "Backtracking"
    global minimum,ansList,nodeCount
    '''CSP graph'''
    graph=constructCSP(L+2,M)
    ''' optimal length'''
    minimum=L+1
    ''' final ruler'''
    ansList=[]
    ''' no of nodes'''
    nodeCount=0
    time1=time.time()
    ''' calling backtrack'''
    backtrack(graph,M,L,0,[0]);
    time2=time.time()
    print "Time Taken :",time2-time1
    print "Nodes Expanded :",nodeCount
    print "Ruler :",ansList[len(ansList)-1],ansList
    if(ansList[len(ansList)-1]<=L):
        return ansList[len(ansList)-1],ansList
    else :
        return -1,[]

#Your backtracking+Forward checking function implementation
def FC(L, M):
    "*** YOUR CODE HERE ***"
    print "Forward Checking"
    global minimum,ansList,nodeCount
    ''' CSP graph '''
    graph=constructCSP(L,M)
    ''' Minimum length'''
    minimum=L+1
    ''' Final Ruler'''
    ansList=[]
    ''' Node count'''
    nodeCount=0
    ''' time taken'''
    time1=time.time()
    ''' calling forward checking'''
    forward(graph,M,L,0,[0],set());
    time2=time.time()
    print "Time Taken :",time2-time1
    print "Nodes Expanded :",nodeCount
    print "Ruler :",ansList[len(ansList)-1],ansList
    if(ansList[len(ansList)-1]<=L):
        return ansList[len(ansList)-1],ansList
    else :
        return -1,[]

#Bonus: backtracking + constraint propagation

def CP(L, M):
    "*** YOUR CODE HERE ***"
    print "Arc Consistent"
    global minimum,ansList,nodeCount
    ''' CSP graph'''
    graph=constructCSP(L,M)
    ''' Making the graph arc-consistent'''
    graph=makeArcConsistent(graph,L,M)
    ''' Minimum value'''
    minimum=L+1
    ''' ans list'''
    ansList=[]
    ''' node count'''
    nodeCount=0
    ''' time taken'''
    time1=time.time()
    ''' calling backtrack after csp optimized with set and minimum maintain'''
    csp_backtrack(graph,M,L,0,[0],set());
    time2=time.time()
    print "Time Taken :",time2-time1
    print "Nodes Expanded :",nodeCount
    print "Ruler :",ansList[len(ansList)-1],ansList
    if(ansList[len(ansList)-1]<=L):
        return ansList[len(ansList)-1],ansList
    else :
        return -1,[]

    return -1,[]
