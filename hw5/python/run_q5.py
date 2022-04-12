import numpy as np
import scipy.io
from nn import *
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import string

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# Q5.1 & Q5.2
# initialize layers here
##########################
##### your code here #####
##########################
N, D = train_x.shape
initialize_weights(D, hidden_size, params, 'layer1')
initialize_weights(hidden_size, hidden_size, params, 'hidden1')
initialize_weights(hidden_size, hidden_size, params, 'hidden2')
initialize_weights(hidden_size, D, params, 'output')

params['m_Wlayer1'] = np.zeros((D, hidden_size))
params['m_blayer1'] = np.zeros(hidden_size)
params['m_Whidden1'] = np.zeros((hidden_size, hidden_size))
params['m_bhidden1'] = np.zeros(hidden_size)
params['m_Whidden2'] = np.zeros((hidden_size, hidden_size))
params['m_bhidden2'] = np.zeros(hidden_size)
params['m_Woutput'] = np.zeros((hidden_size, D))
params['m_boutput'] = np.zeros(D)

# should look like your previous training loops
losses = []
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions

        ##########################
        ##### your code here #####
        ##########################
        h = forward(xb, params, 'layer1', relu)
        h1 = forward(h, params, 'hidden1', relu)
        h2 = forward(h1, params, 'hidden2', relu)
        out = forward(h2, params, 'output', sigmoid)
    
        loss = np.sum(np.square(out - xb))
        total_loss += loss

        delta = 2*(out - xb) 
        delta1 = backwards(delta, params, 'output', sigmoid_deriv)
        delta2 = backwards(delta1, params, 'hidden2', relu_deriv)
        delta3 = backwards(delta2, params, 'hidden1', relu_deriv)
        backwards(delta3, params, 'layer1', relu_deriv)           
        
        params['m_Wlayer1'] = 0.9 * params['m_Wlayer1'] - learning_rate*params['grad_Wlayer1']
        params['Wlayer1'] += params['m_Wlayer1']
        params['m_blayer1'] = 0.9 * params['m_blayer1'] - learning_rate*params['grad_blayer1']
        params['blayer1'] += params['m_blayer1']
        
        params['m_Whidden1'] = 0.9*params['m_Whidden1'] - learning_rate*params['grad_Whidden1']
        params['Whidden1'] += params['m_Whidden1']
        params['m_bhidden1'] = 0.9*params['m_bhidden1'] - learning_rate*params['grad_bhidden1']
        params['bhidden1'] += params['m_bhidden1']
        
        params['m_Whidden2'] = 0.9*params['m_Whidden2'] - learning_rate*params['grad_Whidden2']
        params['Whidden2'] += params['m_Whidden2']
        params['m_bhidden2'] = 0.9*params['m_bhidden2'] - learning_rate*params['grad_bhidden2']
        params['bhidden2'] += params['m_bhidden2']
        
        params['m_Woutput'] = 0.9*params['m_Woutput'] - learning_rate*params['grad_Woutput']
        params['Woutput'] += params['m_Woutput']
        params['m_boutput'] = 0.9*params['m_boutput'] - learning_rate*params['grad_boutput']
        params['b_output'] += params['m_boutput']
    
    losses.append(total_loss/train_x.shape[0])
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9

# plot loss curve
plt.plot(range(len(losses)), losses)
plt.xlabel("epoch")
plt.ylabel("average loss")
plt.xlim(0, len(losses)-1)
plt.ylim(0, None)
plt.grid()
plt.show()

        
# Q5.3.1
# choose 5 labels (change if you want)
visualize_labels = ["A", "B", "C", "1", "2"]

# get 2 validation images from each label to visualize
visualize_x = np.zeros((2*len(visualize_labels), valid_x.shape[1]))
for i, label in enumerate(visualize_labels):
    idx = 26+int(label) if label.isnumeric() else string.ascii_lowercase.index(label.lower())
    choices = np.random.choice(np.arange(100*idx, 100*(idx+1)), 2, replace=False)
    visualize_x[2*i:2*i+2] = valid_x[choices]

# run visualize_x through your network
# name the output reconstructed_x
# both will be plotted below
##########################
##### your code here #####
##########################
h = forward(visualize_x, params, 'layer1', relu)
h1 = forward(h, params, 'hidden1', relu)
h2 = forward(h1, params, 'hidden2', relu)
reconstructed_x = forward(h2, params, 'output', sigmoid)


# plot visualize_x and reconstructed_x
fig = plt.figure()
plt.axis("off")
grid = ImageGrid(fig, 111, nrows_ncols=(len(visualize_labels), 4), axes_pad=0.05)
for i, ax in enumerate(grid):
    if i % 2 == 0:
        ax.imshow(visualize_x[i//2].reshape((32, 32)).T, cmap="Greys")
    else:
        ax.imshow(reconstructed_x[i//2].reshape((32, 32)).T, cmap="Greys")
    ax.set_axis_off()
plt.show()


# Q5.3.2
from skimage.metrics import peak_signal_noise_ratio
# evaluate PSNR
##########################
##### your code here #####
##########################
h = forward(valid_x, params, 'layer1', relu)
h1 = forward(h, params, 'hidden1', relu)
h2 = forward(h1, params, 'hidden2', relu)
reconstructed_x = forward(h2, params, 'output', sigmoid)
total = []
for x, y in zip(valid_x, reconstructed_x):
    total.append(peak_signal_noise_ratio(x.reshape((32, 32)).T, y.reshape((32, 32)).T))
print(np.mean(total))