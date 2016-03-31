import luigi
import sciluigi
import numpy as np
import pandas as pd
import dlib


class Video(sciluigi.ExternalTask):

    corpus_dir = luigi.Parameter(default='/vol/corpora5/mediaeval')
    corpus = luigi.Parameter(default='INA')
    show = luigi.Parameter(default='F2_TS/20130607/130607FR20000_B.MPG')

    def out_put(self):

        INA_TEMPLATE = '{corpus_dir}/INA/LAffaireSnowden/medias_reencoded/tv/{show}.mp4'

        if self.corpus == 'INA':
            path = INA_TEMPLATE.format(
                corpus_dir=self.corpus_dir,
                show=self.show)

        return sciluigi.TargetInfo(self, path)


class Audio(sciluigi.ExternalTask):

    corpus_dir = luigi.Parameter(default='/vol/corpora5/mediaeval')
    corpus = luigi.Parameter(default='INA')
    show = luigi.Parameter(default='F2_TS/20130607/130607FR20000_B.MPG')

    def out_put(self):

        INA_TEMPLATE = '{corpus_dir}/INA/LAffaireSnowden/medias_reencoded/tv/{show}.wav'

        if self.corpus == 'INA':
            path = INA_TEMPLATE.format(
                corpus_dir=self.corpus_dir,
                show=self.show)

        return sciluigi.TargetInfo(self, path)



def _getFaceGenerator(tracking, frame_width, frame_height, double=True):
    """Parse precomputed face file and generate timestamped faces"""

    # load tracking file and sort it by timestamp
    names = ['t', 'track', 'left', 'top', 'right', 'bottom', 'status']
    dtype = {'left': np.float32, 'top': np.float32,
             'right': np.float32, 'bottom': np.float32}
    tracking = pd.read_table(tracking, delim_whitespace=True, header=None,
                             names=names, dtype=dtype)
    tracking = tracking.sort_values('t')

    # t is the time sent by the frame generator
    t = yield

    rectangle = dlib.drectangle if double else dlib.rectangle

    faces = []
    currentT = None

    for _, (T, identifier, left, top, right, bottom, status) in tracking.iterrows():

        left = int(left * frame_width)
        right = int(right * frame_width)
        top = int(top * frame_height)
        bottom = int(bottom * frame_height)

        face = rectangle(left, top, right, bottom)

        # load all faces from current frame and only those faces
        if T == currentT or currentT is None:
            faces.append((identifier, face, status))
            currentT = T
            continue

        # once all faces at current time are loaded
        # wait until t reaches current time
        # then returns all faces at once

        while True:

            # wait...
            if currentT > t:
                t = yield t, []
                continue

            # return all faces at once
            t = yield currentT, faces

            # reset current time and corresponding faces
            faces = [(identifier, face, status)]
            currentT = T
            break

    while True:
        t = yield t, []


def _pairwise(iterable):
    "s -> (s0,s1), (s2,s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)


def _getLandmarkGenerator(shape, frame_width, frame_height):
    """Parse precomputed shape file and generate timestamped shapes"""

    # load landmarks file
    shape = pd.read_table(shape, delim_whitespace=True, header=None)

    # deduce number of landmarks from file dimension
    _, d = shape.shape
    n_points = (d - 2) / 2

    # t is the time sent by the frame generator
    t = yield

    shapes = []
    currentT = None

    for _, row in shape.iterrows():

        T = float(row[0])
        identifier = int(row[1])
        landmarks = np.float32(list(_pairwise(
            [coordinate for coordinate in row[2:]])))
        landmarks[:, 0] = np.round(landmarks[:, 0] * frame_width)
        landmarks[:, 1] = np.round(landmarks[:, 1] * frame_height)

        # load all shapes from current frame
        # and only those shapes
        if T == currentT or currentT is None:
            shapes.append((identifier, landmarks))
            currentT = T
            continue

        # once all shapes at current time are loaded
        # wait until t reaches current time
        # then returns all shapes at once

        while True:

            # wait...
            if currentT > t:
                t = yield t, []
                continue

            # return all shapes at once
            t = yield currentT, shapes

            # reset current time and corresponding shapes
            shapes = [(identifier, landmarks)]
            currentT = T
            break

    while True:
        t = yield t, []
