# The joints definition are copied from [ROMP](https://github.com/Arthur151/ROMP/blob/4eebd3647f57d291d26423e51f0d514ff7197cb3/romp/lib/constants.py#L58)

class Skeleton():
    bones = []
    bone_colors = []
    chains = []
    parent = []

class Skeleton_SMPL24(Skeleton):
    chains = [
        [0, 1, 4, 7, 10],         # left leg
        [0, 2, 5, 8, 11],         # right leg
        [0, 3, 6, 9, 12, 15],     # spine & head
        [12, 13, 16, 18, 20, 22], # left arm
        [12, 14, 17, 19, 21, 23], # right arm
    ]
    bones = [
        [ 0,  1], [ 1,  4], [ 4,  7], [ 7, 10],           # left leg
        [ 0,  2], [ 2,  5], [ 5,  8], [ 8, 11],           # right leg
        [ 0,  3], [ 3,  6], [ 6,  9], [ 9, 12], [12, 15], # spine & head
        [12, 13], [13, 16], [16, 18], [18, 20], [20, 22], # left arm
        [12, 14], [14, 17], [17, 19], [19, 21], [21, 23], # right arm
    ]
    bone_colors = [
        [127,   0,   0], [148,  21,  21], [169,  41,  41], [191,  63,  63],                  # red
        [  0, 127,   0], [ 21, 148,  21], [ 41, 169,  41], [ 63, 191,  63],                  # green
        [  0,   0, 127], [ 15,  15, 143], [ 31,  31, 159], [ 47,  47, 175], [ 63,  63, 191], # blue
        [  0, 127, 127], [ 15, 143, 143], [ 31, 159, 159], [ 47, 175, 175], [ 63, 191, 191], # cyan
        [127,   0, 127], [143,  15, 143], [159,  31, 159], [175,  47, 175], [191,  63, 191], # magenta
    ]
    parent = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 12, 12, 13, 14, 16, 17, 18, 19, 20, 21]
