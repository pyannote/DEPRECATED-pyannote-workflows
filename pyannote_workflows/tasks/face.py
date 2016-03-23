import luigi
import sciluigi

import pickle
import pyannote.core.json
import pyannote.video.face.clustering

from pyannote_workflows.utils import AutoOutput


class PrecomputeClustering(sciluigi.Task, AutoOutput):

    in_openface = None

    def run(self):

        openface = self.in_openface().path

        clustering = pyannote.video.face.clustering.FaceClustering(force=True)
        starting_point, features = clustering.model.preprocess(openface)
        _ = clustering(starting_point, features=features)

        with self.out_put().open('w') as f:
            pickle.dump(clustering.history, f)


class Clustering(sciluigi.Task, AutoOutput):

    in_precomputed = None
    threshold = luigi.FloatParameter()

    def run(self):

        with self.in_precomputed().open('r') as fp:
            history = pickle.load(fp)

        for i, iteration in enumerate(history.iterations):
            if -iteration.similarity > self.threshold:
                break

        result = history[i-1]

        with self.out_put().open('w') as fp:
            pyannote.core.json.dump(result, fp)
