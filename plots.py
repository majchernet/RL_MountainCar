import matplotlib.pyplot as plt
import numpy as np
import settings



"""
Plot actions policy stored in the neural network
"""
def plotPolicyNN(nn,saveToFile="",show=False):
    plt.figure(figsize=(8, 6), dpi=100)
    plt.subplot(1, 1, 1)
    plt.title("Policy defined by neural network")
    
    X = np.linspace(settings.pmin, settings.pmax, settings.dS, endpoint=True)
    Y = np.linspace(settings.vmin, settings.vmax, settings.dS, endpoint=True)
    P = []

    for x in X:
        for y in Y:
            val = nn.ask([[x,y,x*y]])
            a = np.argmax(val)
            P.append([x,y,a])
     
    Xleft = [row[0] for row in P if row[2]==0]
    Yleft = [row[1] for row in P if row[2]==0]
    
    Xwait = [row[0] for row in P if row[2]==1]
    Ywait = [row[1] for row in P if row[2]==1]
    
    Xright = [row[0] for row in P if row[2]==2]
    Yright = [row[1] for row in P if row[2]==2]
    
    # Set x limits
    plt.xlim(settings.pmin, settings.pmax)
    plt.xlabel('Position')
    
    # Set y limits
    plt.ylim(settings.vmin, settings.vmax)
    plt.ylabel('Velocity')
    
          
    plt.plot(Xleft, Yleft, 'b<', label="Move left")        
    plt.plot(Xright, Yright, 'r>', label="Move right")        
    plt.plot(Xwait, Ywait, 'go', label="Wait")

    plt.legend(loc='best', ncol=1)
    
    if saveToFile:
        plt.savefig('{}.png'.format(saveToFile))
    
    if show:
        plt.show()
    
    plt.clf()
    plt.cla()
    plt.close()
    return
    

"""
Plot actions policy

policy - policy that determine action rules
steps  - number of steps after which the goal has been achieved
points - if points have been given, plot decision boundry line 
"""
def plotPolicy(policy,steps,points=[]):
    
    plt.figure(figsize=(8, 6), dpi=100)
    plt.subplot(1, 1, 1)
    plt.title("Policy with the goal achieved after {} time steps".format(steps))
    
    Xleft = [row[0] for row in policy if row[2]==0]
    Yleft = [row[1] for row in policy if row[2]==0]
    
    Xwait = [row[0] for row in policy if row[2]==1]
    Ywait = [row[1] for row in policy if row[2]==1]
    
    Xright = [row[0] for row in policy if row[2]==2]
    Yright = [row[1] for row in policy if row[2]==2]
    
    # Set x limits
    plt.xlim(settings.pmin, settings.pmax)
    plt.xlabel('Position')
    
    # Set y limits
    plt.ylim(settings.vmin, settings.vmax)
    plt.ylabel('Velocity')
    
    plt.plot(Xleft, Yleft, 'b<', label="Move left")        
    plt.plot(Xright, Yright, 'r>', label="Move right")        
    plt.plot(Xwait, Ywait, 'go', label="Wait")
    
    
    if points:
        X = np.linspace(settings.pmin, settings.pmax, settings.dS, endpoint=True)
        # for two points A=(xA,yA), B=(xB,yB)
        # equation of the line passing through two points:
        # Y=(yA−yB)(xA−xB)X+(yA−((yA−yB)/(xA−xB))xA)
        Y = ((points[0][1]-points[1][1])/(points[0][0]-points[1][0]))*X+(points[0][1]-((points[0][1]-points[1][1])/(points[0][0]-points[1][0]))*points[0][0])
        plt.plot(X, Y, color="green", linewidth=3.0, linestyle="-", label="Random decision boundary")        
        plt.plot([points[0][0],points[1][0]], [points[0][1],points[1][1]], 'ko', label="Random points")
    plt.legend(loc='best', ncol=1)
    
    plt.show()
    
    return





"""
Plot actions policy with line decision boundary 

point1, point2 - points that designated line
xRange, yRange - ranges state space
width          - distance from the boundary  in which the action is 0 (no push)
steps          - number of steps after which the goal has been achieved
"""
def spaceDivision(point1, point2, xRange, yRange, width, steps):
    
    plt.figure(figsize=(8, 6), dpi=100)
    plt.subplot(1, 1, 1)
    plt.title("Policy with the goal achieved after {} time steps".format(steps))
    
    # Set x limits
    plt.xlim(xRange[0], xRange[1])
    plt.xlabel('Position')
    
    # Set y limits
    plt.ylim(yRange[0], yRange[1])
    plt.ylabel('Velocity')
    
    X = np.linspace(xRange[0], xRange[1], settings.dS, endpoint=True)
    Y = ((point1[1]-point2[1])/(point1[0]-point2[0])) * X + (point1[1] - ((point1[1]-point2[1])/(point1[0]-point2[0])) * point1[0] )

    
    xleft, yleft, xwait, ywait, xright, yright = [],[],[],[],[],[]
    for x in X:
        for y in np.linspace(yRange[0], yRange[1], 40, endpoint=True):
            p = ((y-point1[1])*(point2[0]-point1[0]) - (point2[1]-point1[1])*(x-point1[0]))
            if p > width:
                xleft.append(x)
                yleft.append(y)
            elif p < -width:
                xright.append(x)
                yright.append(y)
            else:
                xwait.append(x)
                ywait.append(y)
    
    plt.plot(xleft, yleft, 'b<', label="Move left")        
    plt.plot(xright, yright, 'r>', label="Move right")        
    plt.plot(xwait, ywait, 'go', label="Wait")
        
    plt.plot(X, Y, color="green", linewidth=3.0, linestyle="-", label="Random decision boundary")        
    plt.plot([point1[0],point2[0]], [point1[1],point2[1]], 'ko', label="Random points")
    plt.legend(loc='best', ncol=1)
    
    plt.show()
    return


