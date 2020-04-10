from tensorflow.keras.models import load_model
import numpy as np
from keras.preprocessing import image
import cv2

model = load_model("accident_model2.h5")



def Calculate_problilty(filename):
    print("Running...")
    cam = cv2.VideoCapture(filename)
    z = 0
    i = 0
    j = 0
    Action = "Non-Accident"
    newAction = None
    ans = [0, 0]
    while cam.isOpened():
        z = z + 1
        if z % 2 == 0:
            continue
        ret, frame = cam.read()
        if ret == 1:
            frame = cv2.resize(frame, (128, 128))
            x = image.img_to_array(frame) / 255
            x = np.expand_dims(x, axis=0)

            images = np.vstack([x])
            classes = model.predict(images, batch_size=1)
            # print(classes)
            if classes[0] > 0.6:
                newAction = "Accident"
                ans[0] += 1
            else:
                newAction = "Non-Accident"
                ans[1] += 1
            if newAction != Action:
                # print(newAction)
                Action = newAction
            cv2.imshow("video", frame)
            c = cv2.waitKey(1)
            if c == 27:
                break
        else:
            cam.release()
    cam.release()
    cv2.destroyAllWindows()
    prob = ans[0] / (ans[0] + ans[1])
    # print("problility=", prob)
    print("Stoped!")
    return prob


def dijkstra(graph, src, dest, visited=[], distances={}, predecessors={}):

    """ calculates a shortest path tree routed in src
    """
    # a few sanity checks
    if src not in graph:
        raise TypeError('The root of the shortest path tree cannot be found')
    if dest not in graph:
        raise TypeError('The target of the shortest path cannot be found')
        # ending condition
    if src == dest:
        # We build the shortest path and display it
        path = []
        pred = dest
        while pred != None:
            path.append(pred)
            pred = predecessors.get(pred, None)
        # reverses the array, to display the path nicely
        readable = path[0]
        for index in range(1, len(path)): readable = path[index] + '--->' + readable
        # prints it
        # print('shortest path - array: ' + str(path))
        print("path: " + readable + ",   distance=" + str(distances[dest]))
    else:
        # if it is the initial  run, initializes the cost
        if not visited:
            distances[src] = 0
        # visit the neighbors
        for neighbor in graph[src]:
            if neighbor not in visited:
                new_distance = distances[src] + graph[src][neighbor]
                if new_distance < distances.get(neighbor, float('inf')):
                    distances[neighbor] = new_distance
                    predecessors[neighbor] = src
        # mark as visited
        visited.append(src)
        unvisited = {}
        for k in graph:
            if k not in visited:
                unvisited[k] = distances.get(k, float('inf'))
        x = min(unvisited, key=unvisited.get)
        dijkstra(graph, x, dest, visited, distances, predecessors)

if __name__ == "__main__":
    graph = {'a': {'d': 7, 'b': 3, 'f': 15},
             'b': {'c': 1, 'd': 2, 'a': 3},
             'c': {'b': 1, 'd': 2, 'h': 12},
             'd': {'e': 1, 'a': 7, 'c': 2, 'i': 6},
             'e': {'d': 1, 'f': 9, 'h': 6},
             'f': {'a': 15, 'g': 3, 'e': 9, 'h': 8, 's': 5},
             'g': {'f': 3},
             'h': {'f': 8, 'e': 6, 'l': 5, 'p': 1},
             'i': {'d': 6, 'j': 1, 'k': 3},
             'j': {'i': 1, 'l': 3, 'm': 19},
             'k': {'i': 3, 'n': 20},
             'l': {'h': 5, 'q': 6, 'o': 11, 'j': 3},
             'm': {'j': 19, 'o': 1},
             'n': {'k': 20, 'o': 2},
             'o': {'1': 5, 'l': 11, 'm': 1, 'n': 2, 'w': 6},
             'p': {'r': 2, 'l': 6, 'o': 5},
             'q': {'h': 1, 'r': 4},
             'r': {'q': 2, 'p': 4, 't': 7, 'u': 11},
             's': {'t': 2, 'f': 5},
             't': {'r': 7, 's': 2},
             'u': {'r': 11},
             'w': {'o': 6}
             }
    areaName=input("Enter file name: ")
    filname = '{}.mp4'.format(areaName)
    chances = Calculate_problilty(filname)
    print("Probility of accident= ",(chances)*100)
    hospital = 'h'
    if chances > 0.20:
        dijkstra(graph, areaName, hospital)
        print("Alert! Accident happended")
    elif(chances>0.05 and chances<=0.2):
        dijkstra(graph, areaName, hospital)
        print("Warning!")
    else:
        print("No-Accident")