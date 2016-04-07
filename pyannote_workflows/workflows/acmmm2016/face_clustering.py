import luigi
import sciluigi

import pyannote_workflows.tasks.evaluation
import pyannote_workflows.tasks.tvd_dataset
import pyannote_workflows.utils
from pprint import pprint


class _FaceReference(sciluigi.ExternalTask):

    workdir = luigi.Parameter()
    series = luigi.Parameter()
    season = luigi.IntParameter()
    episode = luigi.IntParameter()

    def out_put(self):
        TEMPLATE = '{workdir}/external/face_groundtruth/{episode}.json'
        path = TEMPLATE.format(
            workdir=self.workdir,
            episode=pyannote_workflows.tasks.tvd_dataset.get_episode(self))
        return sciluigi.TargetInfo(self, path)


class _Openface(sciluigi.ExternalTask):

    workdir = luigi.Parameter()
    series = luigi.Parameter()
    season = luigi.IntParameter()
    episode = luigi.IntParameter()

    def out_put(self):
        TEMPLATE = '{workdir}/external/openface/{episode}.txt'
        path = TEMPLATE.format(
            workdir=self.workdir,
            episode=pyannote_workflows.tasks.tvd_dataset.get_episode(self))
        return sciluigi.TargetInfo(self, path)


class FaceClustering(sciluigi.WorkflowTask):

    workdir = luigi.Parameter(default='/work')
    tvddir = luigi.Parameter(default='/tvd')
    series = luigi.Parameter(default='GameOfThrones')
    season = luigi.IntParameter(default=1)
    episode = luigi.IntParameter(default=1)
    language = luigi.Parameter(default='en')

    faceClustering__threshold = luigi.FloatParameter(default=0.4)

    hyperopt = luigi.Parameter(default=None)

    def workflow(self):

        openface = self.new_task(
            'openface',
            _Openface,
            workdir=self.workdir,
            series=self.series,
            season=self.season,
            episode=self.episode)

        precomputeFaceClustering = self.new_task(
            'precomputeFaceClustering',
            pyannote_workflows.tasks.face.PrecomputeClustering)

        precomputeFaceClustering.in_openface = openface.out_put

        faceClustering = self.new_task(
            'faceClustering',
            pyannote_workflows.tasks.face.Clustering,
            threshold=self.faceClustering__threshold)

        faceClustering.in_precomputed = precomputeFaceClustering.out_put

        # =====================================================================
        # EVALUATION
        # =====================================================================

        faceReference = self.new_task(
            'faceReference',
            _FaceReference,
            workdir=self.workdir,
            series=self.series,
            season=self.season,
            episode=self.episode)

        evaluateFaceClustering = self.new_task(
            'evaluateFaceClustering',
            pyannote_workflows.tasks.evaluation.EvaluateDiarizationFast)

        evaluateFaceClustering.in_hypothesis = faceClustering.out_put
        evaluateFaceClustering.in_reference = faceReference.out_put

        if hasattr(self, 'auto_output'):
            pprint(self.auto_output)

        if self.hyperopt is not None:
            hyperopt = self.new_task(
                'hyperopt',
                pyannote_workflows.utils.Hyperopt,
                temp=self.hyperopt)
            hyperopt.in_evaluation = evaluateFaceClustering.out_put
            return hyperopt

        else:
            return evaluateFaceClustering
