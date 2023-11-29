
def distance_boxes (boxA, boxB):
    import math
    center_boxA = [(boxA[0] + boxA[2])/ 2.0, (boxA[1] + boxA[3])/2.0]
    center_boxB = [(boxB[0] + boxB[2])/ 2.0, (boxB[1] + boxB[3])/2.0]
    pixel_distance  = math.sqrt( ((center_boxA[0]-center_boxB[0])**2)+((center_boxA[1]-center_boxB[1])**2) )
    return pixel_distance
 