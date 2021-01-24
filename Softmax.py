import numpy as np
import math

np.random.seed(100)
trainImg_dif = "train_img.txt"
trainLab_dif = "train_label.txt"
testImg_dif = "test_img.txt"
testLab_dif = "test.txt"

train_ratio = 0.8
learning_rate = 0.5
Epoch = 2
tau = 0.001
output_dim = 3

Layer = [8, 16, 16, 8]

# read data from file
with open(trainImg_dif) as f:
    train_data = np.array([line.strip().split(',') for line in f],int)

train_label = np.zeros( ( len(train_data), output_dim ) )
with open(trainLab_dif) as f:
    tmp = np.array([line.strip().split() for line in f],int)
for i in range (len(tmp) ):
    train_label[i][tmp[i]] = 1

with open(testImg_dif) as f:
    test_data = np.array([line.strip().split(',') for line in f],int)

# add input dim and output dim
Layer.insert(0, len(train_data[0]))
Layer.append(output_dim)


def initialize(Layer):
    Weights = []
    Bias = []

    for i in range ( 1, len(Layer) ):
        weight = np.random.randn( Layer[i-1], Layer[i]) * np.sqrt(2 / Layer[i])
        b = np.random.randn(Layer[i], 1) * np.sqrt(2 / Layer[i])
        Weights.append(weight)
        Bias.append(b)

    # to make index of layer 1 from 0 to 1
    Weights.insert(0, [])
    Bias.insert(0, [])
    Weights = np.asarray(Weights)
    Bias = np.asarray(Bias)

    return Weights, Bias

def Partition(data, label, ratio):
    t_data = Minmax_scale( data[0:int(len(data)*ratio) ], 0, 255 )
    t_label = label[0:int(len(label)*ratio) ]
    verify_data = Minmax_scale( data[int(len(data)*ratio):len(data)], 0, 255 )
    verify_label = label[int(len(data)*ratio):len(data)]
    return t_data, t_label, verify_data, verify_label

def Sigmoid( n ):
    l = []
    for tmp in n:
        l.append( 1 / (1 + math.exp(-tmp) ) )
    return l

def Binary_crossEntropy(y, a):
    CE = 0
    for i in range( len(y) ):
        CE += y[i] * math.log(a[i] + 1e-15 ) + (1 - y[i]) * math.log(1 - a[i] + 1e-15)
    return CE

def Prediction(a):
    index = 0
    for i in range( len(a) ):
        if a[i] > a[index]:
            index = i
    return index

def Minmax_scale(w, min, max):
    return np.array( (w-min) / (max-min) )

def FeedForward(data, Weight, bias):
    n = np.matmul( data, Weight) + bias.transpose() 
    return np.asarray( Sigmoid(n[0]) ) 

def softmax( n ):
    n = np.asarray(n)
    t = np.exp(n)/sum(np.exp(n))
    return t
        
def CrossEntropy(y, a):
    CE = 0
    for j in range( len(y) ):
        CE += y[j] * math.log(a[j] + 1e-15)
    return CE

Weights, Bias = initialize(Layer)
t_data, t_label, verify_data, verify_label = Partition(train_data, train_label, train_ratio)

end_flag = False
last_acc_T = 0
last_acc_V = 0
ce = []
for ep in range(Epoch):
    totalLoss = 0
    L = len(Layer)-1
    acc_t = 0
    acc_v = 0
    
    flag = True
    for i in range(len(t_data)):
        a = []
        a.append( t_data[i] )
        for l in range(1, L ):
            a.append(FeedForward(a[l-1],  Weights[l], Bias[l]) )
        
        # do softmax output layer 
        n = np.matmul( a[L-1], Weights[L]) + Bias[L].transpose() 
        a.append( softmax(n[0]) )

        # BackWard
        
        Delta = [ a[L] - t_label[i] ]
        Delta[0] /= output_dim

        for l in range(L-1, 0, -1):
            Delta.insert(0, np.matmul( Delta[0], Weights[l+1].transpose() ) * ( a[l]*(1-a[l])  )  )
        Delta.insert(0, [])
        
        # update Weights and Bias
        for l in range(1, L):
            t = np.matmul( Delta[l][np.newaxis].T, a[l-1][np.newaxis] )
            Weights[l] = Weights[l] - learning_rate * t.T
            if l == L:
                break
            Bias[l] = Bias[l] - learning_rate * Delta[l]
        
        Bias[L] += -learning_rate * np.sum(Delta[L])

        # Calculate loss and training accuracy
        if flag : 
            totalLoss -= CrossEntropy( t_label[i], a[L] )
            p = Prediction(a[L])
            if t_label[i][p] == 1:
                acc_t += 1

    # Calculate verify accuracy
    if flag : 
        for i in range( len(verify_data) ):

            out = verify_data[i]
            for l in range(1, L+1 ):
                out = FeedForward(out, Weights[l], Bias[l])

            p = Prediction(out)
            if verify_label[i][p] == 1:
                acc_v += 1

        print("Epoch :" + str(ep+1) )
        print( str( totalLoss / len(t_data) ) )
        print( "train accuracy :" + str(acc_t / len(t_data)) )
        print( "verify accuracy :" + str(acc_v / len(verify_data)) )

        if last_acc_T < acc_t / len(t_data) and last_acc_V > acc_v / len(verify_data):
            print("< Overtraining leads to overfitting >" )
            end_flag = True

        last_acc_T = acc_t
        last_acc_V = acc_v
    if (ep+1) == Epoch:
        print("< Reach maximum Epoch >" )
        end_flag = True

    if flag and totalLoss / len(t_data) < tau:
        print("< Loss is low enough ( tau: " + str(tau) + ") >"  )
        end_flag = True

    if end_flag:
        print("Number of train_data: " + str( len(t_data)) )
        print("Number of verify_data: " + str( len(verify_data)) )
        print("Number of Hidden Layer: " + str( len(Layer)-2) )
        print("Number of Neuron in each Hidden Layer: " + str(Layer[1:len(Layer)-1]) )
        print("Learning Rate: " + str(learning_rate) )
        print("Epoch: " + str(ep+1) )
        print("Loss: " + str( totalLoss / len(t_data) ))
        print("Train accuracy: " + str(acc_t / len(t_data)) )
        print("Verify accuracy: " + str(acc_v / len(verify_data)) )
        break

# predict test data
test_prediction = []
for data in test_data:
    out = Minmax_scale(data, 0, 255)
    for l in range(1, L+1 ):
        out = FeedForward(out, Weights[l], Bias[l])
    test_prediction.append( Prediction(out) )

np.savetxt(testLab_dif, np.asarray(test_prediction, dtype = int), fmt="%d")

