from time import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
import random as rand
import numpy as np
import copy

import matplotlib.pyplot as plt
import matplotlib

import io
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

NUM_CLASSES = 5

def build_model(grid_size):
    model = keras.Sequential([
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
    ])
    
    model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy', tf.keras.metrics.categorical_accuracy])


    model.summary()
    return model

AGENT = 3
GOAL = 2
WALL = 1
EMPTY = 0
FIRST = 4

NO_DIR = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4

DIRECTION_MAP = {UP : 'UP', DOWN: 'DOWN', LEFT: 'LEFT', RIGHT : 'RIGHT', NO_DIR:'NONE'}
DIRECTION_DELTA = {UP: (0, 1), DOWN: (0, -1), LEFT: (-1, 0), RIGHT:(1, 0), NO_DIR: (0, 0)}
MAP_CHARS = {'X':WALL, ' ':EMPTY, 'G':GOAL, 'A': AGENT}

class Map:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.best_path = None
        self.map = [[0 for y in range(0, height)] for x in range(0, width)]

    def randomize(self, percent_walls, wall_surround):
        start = 0
        end_width = self.width
        end_height = self.height
        if wall_surround:
            start = 1
            end_width -= 1
            end_height -= 1
            for i in range(0, self.height):
                self.map[0][i] = WALL
                self.map[self.width - 1][i] = WALL
            for i in range(0, self.width):
                self.map[i][0] = WALL
                self.map[i][self.height - 1] = WALL
        self.best_path = None
        self.goal = [rand.randrange(start, end_width), rand.randrange(start, end_height)]
        self.agent = self.goal.copy()
        while ((self.agent[0] == self.goal[0]) and (self.agent[1] == self.goal[1])):
            self.agent = [rand.randrange(start, end_width), rand.randrange(start, end_height)]
        self.map[self.agent[0]][self.agent[1]] = AGENT
        self.map[self.goal[0]][self.goal[1]] = GOAL
        if wall_surround:
            num_walls = int(((self.width-2) * (self.height-2)) * percent_walls)
        else:
            num_walls = int((self.width * self.height) * percent_walls)
        while num_walls > 0:
            x = rand.randrange(start, end_width)
            y = rand.randrange(start, end_height)
            if self.map[x][y] == EMPTY:
                self.map[x][y] = WALL
                num_walls -= 1

    def randomize_from_proto(self, proto_map):
        # randomly swap N% of the squares
        self.width = proto_map.width
        self.height = proto_map.height
        self.goal = proto_map.goal
        self.agent = proto_map.agent
        self.map = copy.deepcopy(proto_map.map)
        for _ in range(0, int((self.width-1) * (self.height-1) * 0.05)):
            x = rand.randrange(1, self.width - 1)
            x2 = rand.randrange(1, self.width - 1)
            y = rand.randrange(1, self.height - 1)
            y2 = rand.randrange(1, self.height - 1)
            if x != x2 or y != y2:
                save = self.map[x][y]
                self.map[x][y] = self.map[x2][y2]
                self.map[x2][y2] = save
                if self.map[x][y] == GOAL:
                    self.goal = [x, y]
                if self.map[x2][y2] == GOAL:
                    self.goal = [x2, y2]
                if self.map[x][y] == AGENT:
                    self.agent = [x,y]
                if self.map[x2][y2] == AGENT:
                    self.agent = [x2, y2]

            
    
    def is_passable(self, pos):
        return self.map[pos[0]][pos[1]] != WALL

    def passable_neighboors(self, node):
        out = []
        if node[0] > 0:
            out.append((node[0] - 1, node[1]))
        if node[1] > 0:
            out.append((node[0], node[1] - 1))
        if node[0] < (self.width - 1):
            out.append((node[0] + 1, node[1]))
        if node[1] < (self.height - 1):
            out.append((node[0], node[1] + 1))
        return [x for x in out if self.is_passable(x)]

    def find_best_path(self):
        if self.best_path:
            return self.best_path
        pos = (self.agent[0], self.agent[1])
        goal = (self.goal[0], self.goal[1])
        open_nodes = [(pos, 0, [])]
        distances = {pos: 0}
        while open_nodes:
            node = open_nodes.pop(0)
            next = self.passable_neighboors(node[0])
            new_distance = node[1] + 1
            for n in next:
                new_path = node[2].copy()
                new_path.append(n)
                if n == goal:
                    new_path = node[2].copy()
                    new_path.append(n)
                    self.best_path = new_path
                    return new_path
                if n not in distances or new_distance < distances[n]:
                    distances[n] = new_distance
                    open_nodes.append((n, new_distance, new_path))

    def compute_naive_first_step(self):
        diff = [self.goal[0] - self.agent[0], self.goal[1] - self.agent[1]]
        if (abs(diff[0]) > abs(diff[1])):
            return RIGHT if (diff[0] > 0) else LEFT
        else:
            return UP if (diff[1] > 0) else DOWN

    def compute_naive_first_step_dirs(self):
        diff = [self.goal[0] - self.agent[0], self.goal[1] - self.agent[1]]
        return [1 if (diff[0] > 0) else -1, 0], [0, 1 if (diff[1] > 0) else -1]

    def path_first_step_dir(self):
        path = self.find_best_path()
        if not path:
            return NO_DIR
        if path[0][0] > self.agent[0]:
            return RIGHT
        elif path[0][0] < self.agent[0]:
            return LEFT
        elif path[0][1] > self.agent[1]:
            return UP
        else:
            return DOWN

    def move_agent(self, dir):
        if not dir or dir == NO_DIR:
            return
        new_loc = [self.agent[0], self.agent[1]]
        dir_delta = DIRECTION_DELTA[dir]
        new_loc[0] += dir_delta[0]
        new_loc[1] += dir_delta[1]
        return self.move_agent_to(new_loc)

    def move_agent_to(self, new_loc):
        self.map[self.agent[0]][self.agent[1]] = EMPTY
        prev_value = self.map[new_loc[0]][new_loc[1]]
        if prev_value == EMPTY or prev_value == GOAL:
            self.map[new_loc[0]][new_loc[1]] = AGENT
            self.agent = new_loc
            self.best_path = None
            return True
        else:
            return False

    def to_string(self, show_path=False):
        width = len(self.map)
        height = len(self.map[0])
        best_path = None
        lines = []
        if show_path:
            best_path = self.find_best_path()
        for y in range(height - 1, -1, -1):
            line = ""
            for x in range(0, width):
                cell = self.map[x][y]
                if cell == EMPTY:
                    if best_path and (x, y) in best_path:
                        line += "*"
                    else:
                        line += " "
                elif cell == WALL:
                    line += "X"
                elif cell == GOAL:
                    line += "G"
                elif cell == AGENT:
                    line += "A"
            lines.append(line)
        return lines

    def print(self):
        width = len(self.map)
        height = len(self.map[0])
        best_path = self.find_best_path()
        for line in self.to_string(show_path=True):
            print(line)
        print("Agent: " + str(self.agent) + " Goal: " + str(self.goal))
        print(str(best_path))
        print("Hard Path: " + str(hard_path(self, best_path)))
        if best_path:
            print("Path Starts Diff Dir: " + str(path_starts_diff_dir(self, best_path)))
            print("Direction Changes: " + str(path_direction_changes(best_path)))

    def plot_map_with_prediction(self, predictions_array, label):
        cmap = matplotlib.colors.ListedColormap(["white","black",'green','red'], name='from_list', N=None)
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(np.array(self.map).transpose(), cmap=cmap, origin="lower")
        plt.grid(False)
        plt.subplot(1,2,2)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        thisplot = plt.bar(range(NUM_CLASSES), predictions_array, color="#777777")
        plt.ylim([0, 1]) 
        plt.xticks(range(0, 5), ['NONE', 'UP', 'DOWN', 'LEFT', 'RIGHT'])
        predicted_label = np.argmax(predictions_array)
        thisplot[predicted_label].set_color('red')
        thisplot[label].set_color('blue')

    def draw_map_with_prediction(self, predictions_array, label):
        self.plot_map_with_prediction(predictions_array, label)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return buf

    def save_map_with_prediction(self, predictions_array, label, filename):
        self.plot_map_with_prediction(predictions_array, label)
        plt.savefig(filename, format='png')

    def save(self, filename):
        print("Saving map to " + filename)
        with open(filename, 'w') as fp:
            fp.writelines([line + "\n" for line in self.to_string(show_path=False)])

    def load(self, filename):
        lines = []
        with open(filename, 'r') as fp:
            for line in fp:
                lines.append(line)
        self.height = len(lines)
        self.width = len(lines[0].strip())
        self.best_path = None
        self.map = [[0 for y in range(0, self.height)] for x in range(0, self.width)]
        for y in range(0, self.height):
            line = lines[(self.height - 1) - y]
            for x in range(0, self.width):
                self.map[x][y] = MAP_CHARS[line[x]]
                if line[x] == 'G':
                    self.goal = [x, y]
                elif line[x] == 'A':
                    self.agent = [x, y]
        print("Loaded:")
        self.print()
        


def path_direction_changes(path):
    num_changes = 0
    last_dir = (0, 0)
    for i in range(0, len(path)-1):
        dir = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
        if dir != last_dir:
            num_changes += 1
        last_dir = dir
    return num_changes

def path_starts_diff_dir(map, path):
    start_dir = (path[0][0] - map.agent[0], path[0][1] - map.agent[1])
    naive_dir = map.compute_naive_first_step_dirs()
    return (naive_dir[0][0] != start_dir[0] or naive_dir[0][1] != start_dir[1]) and \
        (naive_dir[1][0] != start_dir[0] or naive_dir[1][1] != start_dir[1])

def hard_path(map, path):
    if path:
        if path_starts_diff_dir(map, path):
            return True
        else:
            return path_direction_changes(path) > 4 and len(path) >= map.width
    else:
        return False

def generateLongPathStepsBatch(batchSize, grid_size, min_length, min_segment_sz=1, proto_map=None):
    print("Generating Long Path Steps Batch...")
    maps = []
    gen = 0
    gen_good = 0
    while len(maps) < batchSize:
        gen += 1
        map = Map(grid_size, grid_size)
        if proto_map:
            map.randomize_from_proto(proto_map)
            #map.print()
        else:
            map.randomize(0.50, True)
        path = map.find_best_path()
        if not path or len(path) < min_length or not hard_path(map, path):
            continue
        gen_good += 1
        # Generate a copy of the map with every point along the path
        while len(path) > min_segment_sz:
            next_map = copy.deepcopy(map)
            next_map.move_agent_to(path[0])
            maps.append(next_map)
            path = next_map.find_best_path()
    print("Generated Long Path Steps {0}/{1}".format(gen_good, gen))
    return maps
        
def generateHardBatch(batchSize, grid_size, min_length, num_allow_empty=0):
    maps = []
    num_empty = 0
    while len(maps) < batchSize:
        map = Map(grid_size, grid_size)
        map.randomize(0.50, True)
        path = map.find_best_path()
        path_length = 0
        if path:
            path_length = len(path)
        if (num_empty < num_allow_empty or (path and path_length >= min_length and hard_path(map, path))):
            if not path:
                num_empty += 1
            maps.append(map)
    return maps

def generateBatch(batchSize, grid_size, num_allow_empty=0, min_length=0):
    maps = []
    num_empty = 0
    
    while len(maps) < batchSize:
        map = Map(grid_size, grid_size)
        map.randomize(0.50, True)
        path = map.find_best_path()
        path_length = 0
        if path:
            path_length = len(path)
        if (num_empty < num_allow_empty or path) and path_length >= min_length:
            if not path:
                num_empty += 1
            maps.append(map)
    return maps

def createTrainingSet(maps):
    map_tensors = []
    map_labels = []
    for map in maps:
        map_tensors.append(keras.utils.to_categorical(np.array(map.map), 4))
        map_labels.append(map.path_first_step_dir())
    xs = np.array(map_tensors)
    labels = keras.utils.to_categorical(map_labels)
    return xs, labels

def print_status(labels, maps, predictions, log_dir):
    print("Num Predictions: " + str(len(predictions)))
    correct = 0
    writer = tf.summary.FileWriter(log_dir)
    for i in range(0, len(labels)):
        with tf.Session() as sess:
            #print_map(examples['xs'].eval(session=sess)[i])
            map = maps[i]
            map.print()
            label = np.argmax(labels[i])
            prediction = np.argmax(predictions[i])
            plot_buf = maps[i].draw_map_with_prediction(predictions[i], label)

            # Convert PNG buffer to TF image
            image = tf.image.decode_png(plot_buf.getvalue(), channels=4)

            # Add the batch dimension
            image = tf.expand_dims(image, 0)

            # Add image summary
            summary_op = tf.summary.image("plot", image)
            summary = sess.run(summary_op)
            # Write summary

            writer.add_summary(summary)
            
            print("Label: " + DIRECTION_MAP[label])
            print("Pred : " + DIRECTION_MAP[prediction])
            print("Match? " + ("Yes" if prediction == label else "No"))
            if prediction == label:
                correct += 1
            else:
                print(str(predictions[i]))
    print("Performance = %d/%d" % (correct, len(labels)))
    writer.close()

def train(start_model, grid_size, callbacks, log_dir, num_batches, batch_sz, epochs, eval_sz, load_map_filename):
    if start_model:
        model = start_model
    else:
        model = build_model(grid_size)

    proto_map = None
    if load_map_filename:
        proto_map = Map(grid_size,grid_size)
        proto_map.load(load_map_filename)

    #validation_batch = createTrainingSet(generateBatch(eval_sz, grid_size, num_allow_empty=int(eval_sz / 2)))
    validation_batch = createTrainingSet(generateLongPathStepsBatch(eval_sz, grid_size, grid_size, proto_map=proto_map, min_segment_sz=10))

    for i in range(0, num_batches):
        print("Training Batch " + str(i) + "/" + str(num_batches))
        #batch = createTrainingSet(generateBatch(batch_sz, grid_size, num_allow_empty=int(batch_sz / 10)))
        batch = createTrainingSet(generateLongPathStepsBatch(batch_sz, grid_size, grid_size, proto_map=proto_map, min_segment_sz=10))
        history = model.fit(batch[0],
                            batch[1],
                            epochs=epochs,
                            validation_data=(validation_batch[0], validation_batch[1]),
                            batch_size=128,
                            verbose=2,
                            callbacks=callbacks)
    
    return model

def evaluate(model, grid_size, log_dir, eval_sz, proto_map=None):
    model.summary()
    validation_batch = createTrainingSet(generateBatch(eval_sz, grid_size, num_allow_empty=int(eval_sz / 2)))
    solvable_validation_batch = createTrainingSet(generateBatch(eval_sz, grid_size, num_allow_empty=0))
    hard_maps = generateHardBatch(eval_sz, grid_size, min_length=0)
    hard_batch = createTrainingSet(hard_maps)
    long_maps = createTrainingSet(generateLongPathStepsBatch(eval_sz, grid_size, grid_size, min_segment_sz=10))

    #predictions = model.predict(hard_batch[0], steps=50)
    #print_status(hard_batch[1], hard_maps, predictions, log_dir)

    test_loss, test_acc, cat_acc = model.evaluate(validation_batch[0], validation_batch[1])
    print('Test accuracy:', test_acc)
    test_loss, test_acc, cat_acc = model.evaluate(solvable_validation_batch[0], solvable_validation_batch[1])
    print('Test Solvable accuracy:', test_acc)
    test_loss, test_acc, cat_acc = model.evaluate(hard_batch[0], hard_batch[1])
    print('Test Solvable Hard accuracy:', test_acc)
    test_loss, test_acc, cat_acc = model.evaluate(long_maps[0], long_maps[1])
    print('Test Solvable Long accuracy:', test_acc)

def draw_navigation(model, grid_size, load_map_filename=None, save_map_filename=None):
    print("Navigate...")
    map = None
    if load_map_filename:
        map = Map(grid_size,grid_size)
        map.load(load_map_filename)
    else:
        hard_maps = generateBatch(1, grid_size, num_allow_empty=0, min_length=grid_size)
        map = hard_maps[0]
    if save_map_filename:
        map.save(save_map_filename)

    frame_num = 0
    while map.agent[0] != map.goal[0] or map.agent[1] != map.goal[1]:
        prediction = predict(map, model)[0]
        dir = np.argmax(prediction)
        map.print()
        map.save_map_with_prediction(prediction, map.path_first_step_dir(), "img/navigate-" + str(frame_num) + ".png")
        if not dir or dir == NO_DIR:
            print("Path finding failed!")
            break
        print("Attempting to move " + DIRECTION_MAP[dir])
        move_success = map.move_agent(dir)
        if not move_success:
            print("Failed to move " + DIRECTION_MAP[dir])
            break
        print("Dir: " + DIRECTION_MAP[dir])
        print("Pred: " + str(prediction))
        
        frame_num += 1

def predict(map, model):
    prediction = model.predict(np.array([keras.utils.to_categorical(np.array(map.map), 4)]))
    print(str(prediction))
    return prediction