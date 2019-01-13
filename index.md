## Deep Learning for grid based Pathfinding


### Background

Pathfinding!  Pathfinding is a favorite problem of mine from back in my game programming days.  There are standard algorithms to use including the most common [Dijkstra's](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm) and [A*](https://en.wikipedia.org/wiki/A*_search_algorithm).  I have implemented these many times for games, with different map representations.

The simplest map representation has been the old standard grid with spaces a walls.

![Image of Board](/docimg/fullpath.gif)

In this picture black squares represents walls.  White squares are empty.  Red squares are the starting point for our agent, and the green square is the goal the agent is trying to reach.

### Deep Learning Approach

Intuitively if you look at a grid map, you can often immediately "see" the path without having to iterate through a set of spaces as the common algorithms do.  So, is it possible to build a deep learning model to "see" the path?  Let's investigate.

I took a lot of inspiration from [this paper](https://www.sciencedirect.com/science/article/pii/S1877050918300553) although I haven't done any of the cool reinforcement parts.

Defining our problem:

```markdown
Input: An NxM matrix representing the map.  Matrix entires are one hot vectors indicating empty, wall, agent, or goal.
Output: Best best direction to move first.
```

The nice thing about definining the problem this way is that it is easy to generate training examples.  We just need to create some maps, run a traditional pathfinding algorithm to find the full path, and then use the first step of that as a label for the example.

I started with 10x10 maps, but this doesn't let us get into interesting examples where greedy search doesn't work.

### Model Design

Pathfinding centers around analysis of individual cells and their neighboors.  Traditional pathfinding involves visiting neighboors from the start point and exploring outward.  It intuitively makes sense that a convolution filter as the start of our model would be successful as these convolution layers would be better able to represent the neighboor hoods of nodes on our grid and the associated spatial connectivity.

I experimented with a few different layering approachs, but settled on a fairly standard setup that came from some of the standard MNIST hardwriting recognition models.  This layering approach produces far fewer parameters than having just a few fully connected dense layers.

```
keras.layers.Conv2D(grid_size + 2, kernel_size=5, padding="Same", input_shape=[grid_size, grid_size, 4], activation='relu'),
keras.layers.Conv2D(grid_size + 2, kernel_size=5, padding="Same", activation='relu'),
keras.layers.MaxPooling2D(pool_size=(2, 2)),
keras.layers.Dropout(0.25),

keras.layers.Conv2D((grid_size + 2) * 2, kernel_size=3, padding="Same", activation='relu'),
keras.layers.Conv2D((grid_size + 2) * 2, kernel_size=3, padding="Same", activation='relu'),
keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
keras.layers.Dropout(0.25),

keras.layers.Flatten(),
keras.layers.Dense((grid_size * grid_size)/2, activation=tf.nn.relu),
keras.layers.Dropout(0.25), 
keras.layers.Dense(NUM_CLASSES, activation=tf.nn.softmax)
```

I didn't do a lot of exploration here beyond taking off the shelf ideas.  In the future I'd like to do some work on hyperparameter optimization here.

### Generating Training Examples

I started with a simple approach to generating maps: randomly select cells a percentage of cells to be walls vs. open and then randomly select the start and end points.  This has a high probability of creating unsolvable maps (more on that latter).  

The problem with this approach is that it tends to create lots of training examples that are fairly easy.  We want our model to learn to actually find paths through complex mazes, not just greedily move towards the goal. 

To this end, we add additional tests to each of our generated maps and throw out ones that aren't interesting:
- Longer paths are generally more interesting.  Setting a threshold equal to the length of one side of the map seems to work.
- Choose paths where the first move is not in the obvious greedy direction.  I.e. if the goal is up and to the left, choose maps where the path starts down or to the right.

### Tensorflow Keras 

As you guessed from above, I used Keras to define the layers for my model and then Tensorflow to run the training.
This is the parameter space for a 22x22 map.  340k parameters is a lot!

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 22, 22, 24)        2424
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 22, 22, 24)        14424
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 11, 11, 24)        0
_________________________________________________________________
dropout (Dropout)            (None, 11, 11, 24)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 11, 11, 48)        10416
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 11, 11, 48)        20784
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 48)          0
_________________________________________________________________
dropout_1 (Dropout)          (None, 5, 5, 48)          0
_________________________________________________________________
flatten (Flatten)            (None, 1200)              0
_________________________________________________________________
dense (Dense)                (None, 242)               290642
_________________________________________________________________
dropout_2 (Dropout)          (None, 242)               0
_________________________________________________________________
dense_1 (Dense)              (None, 5)                 1215
=================================================================
Total params: 339,905
Trainable params: 339,905
Non-trainable params: 0
_________________________________________________________________
```

### Visualizing Results

Using matplotlib to draw the maps turned out to be easier than I expected:

```
cmap = matplotlib.colors.ListedColormap(["white","black",'green','red'], name='from_list', N=None)
plt.imshow(np.array(self.map).transpose(), cmap=cmap, origin="lower")
```
I had to transpose the map as my coordinate system and layout of the map arrays doesn't quite match what matplotlib wants.

In addition to showing the map, we also want to visualize the outputs of the model.  I used an example from the [tensorflow docs](https://www.tensorflow.org/tutorials/keras/basic_classification) that shows the correct answer as blue and the predicted answer as red if it is wrong.

We can then combine these together into a single plot of the current map state and the agent's decision making output:

![Image of Board](/docimg/mapandprediction.png)

### Training on Google Cloud ML Engine

### Learning to recognize "No Path" 

## Inference

Now that we have a trained model, let's use it!  This part is pretty straight forward.

1.  Create map
2.  Predict direction with model on map
3.  Update map to move agent to in the direction choice.
4.  Repeat prediction and goal until we meet our goal or something fails.

![Image of Board](/docimg/fullpath2.gif)

## Failure Modes

### Oscilation

There are places where the system clearly fails. Because the model only considers a single move at a time, it can get stuck in an oscilating pattern where it moves to a spot and then wants to move back to the previous stop.

![Image of Board](/docimg/oscilate.gif)

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/tstanis/learn_map/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
