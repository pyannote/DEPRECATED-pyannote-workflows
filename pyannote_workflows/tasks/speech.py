import luigi
import sciluigi

from pyannote_workflows.utils import AutoOutput

import pickle
import pyannote.core.json

import pyannote.algorithms.segmentation.bic
import pyannote.algorithms.clustering.bic
import pyannote.features.audio.yaafe
import pyannote.algorithms.segmentation.hmm

from pyannote.core import Segment, Timeline, Annotation


class MFCC(sciluigi.Task, AutoOutput):

    in_audio = None

    e = luigi.BoolParameter()
    De = luigi.BoolParameter(default=True)
    DDe = luigi.BoolParameter(default=True)

    coefs = luigi.IntParameter(default=11)
    D = luigi.BoolParameter(default=True)
    DD = luigi.BoolParameter(default=True)

    def run(self):

        # TODO / compute sample rate and co. based on in_audio
        extractor = pyannote.features.audio.yaafe.YaafeMFCC(
            sample_rate=16000, block_size=512, step_size=256,
            e=self.e, De=self.De, DDe=self.DDe,
            coefs=self.coefs, D=self.D, DD=self.DD)

        mfcc = extractor.extract(self.in_audio().path)

        with self.out_put().open('w') as fp:
            pickle.dump(mfcc, fp)


class BICSegmentation(sciluigi.Task, AutoOutput):

    in_segmentation = None
    in_features = None

    penalty_coef = luigi.FloatParameter(default=1.0)
    covariance_type = luigi.Parameter(default='full')
    min_duration = luigi.FloatParameter(default=1.0)
    precision = luigi.FloatParameter(default=0.1)

    def run(self):

        segmenter = pyannote.algorithms.segmentation.bic.BICSegmentation(
            penalty_coef=self.penalty_coef,
            covariance_type=self.covariance_type,
            min_duration=self.min_duration,
            precision=self.precision)

        with self.in_features().open('r') as fp:
            features = pickle.load(fp)

        with self.in_segmentation().open('r') as fp:
            segmentation = pyannote.core.json.load(fp)

        timeline = segmenter.apply(features, segmentation=segmentation)

        annotation = Annotation()
        for s, segment in enumerate(timeline):
            annotation[segment] = s

        with self.out_put().open('w') as fp:
            pyannote.core.json.dump(annotation, fp)


class LinearBICClustering(sciluigi.Task, AutoOutput):

    in_segmentation = None
    in_features = None

    penalty_coef = luigi.FloatParameter(default=1.0)
    covariance_type = luigi.Parameter(default='diag')

    def run(self):

        clustering = pyannote.algorithms.clustering.bic.LinearBICClustering(
            penalty_coef=self.penalty_coef,
            covariance_type=self.covariance_type)

        with self.in_features().open('r') as fp:
            features = pickle.load(fp)

        with self.in_segmentation().open('r') as fp:
            starting_point = pyannote.core.json.load(fp)

        result = clustering(starting_point, features=features)

        with self.out_put().open('w') as fp:
            pyannote.core.json.dump(result, fp)


class BICClustering(sciluigi.Task, AutoOutput):

    in_segmentation = None
    in_features = None

    penalty_coef = luigi.FloatParameter(default=3.5)
    covariance_type = luigi.Parameter(default='full')

    def run(self):

        clustering = pyannote.algorithms.clustering.bic.BICClustering(
            penalty_coef=self.penalty_coef,
            covariance_type=self.covariance_type)

        with self.in_features().open('r') as fp:
            features = pickle.load(fp)

        with self.in_segmentation().open('r') as fp:
            starting_point = pyannote.core.json.load(fp)

        result = clustering(starting_point, features=features)

        with self.out_put().open('w') as fp:
            pyannote.core.json.dump(result, fp)


class TrainResegmentation(sciluigi.Task, AutoOutput):

    in_segmentation = None
    in_features = None

    n_components = luigi.FloatParameter(default=64)
    covariance_type = luigi.Parameter(default='diag')
    calibration = luigi.Parameter(default='isotonic')
    equal_priors = luigi.BoolParameter(default=True)

    def run(self):

        segmenter = pyannote.algorithms.segmentation.hmm.GMMSegmentation(
            n_jobs=1,  # n_jobs > 1 will fail (not sure why)
            n_iter=10, lbg=True,
            n_components=self.n_components,
            calibration=self.calibration,
            equal_priors=self.equal_priors)

        with self.in_features().open('r') as fp:
            features = pickle.load(fp)

        with self.in_segmentation().open('r') as fp:
            segmentation = pyannote.core.json.load(fp)

        segmenter.fit([features], [segmentation])

        with self.out_put().open('w') as fp:
            pickle.dump(segmenter, fp)


class ApplyResegmentation(sciluigi.Task, AutoOutput):

    in_model = None
    in_features = None

    min_duration = luigi.FloatParameter(default=0.5)

    def run(self):

        with self.in_model().open('r') as fp:
            segmenter = pickle.load(fp)

        with self.in_features().open('r') as fp:
            features = pickle.load(fp)

        segmentation = segmenter.predict(
            features, min_duration=self.min_duration)

        with self.out_put().open('w') as fp:
            pyannote.core.json.dump(segmentation, fp)
