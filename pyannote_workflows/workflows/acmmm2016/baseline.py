import luigi
import sciluigi

import pyannote_workflows.tasks.speech
import pyannote_workflows.tasks.evaluation
import pyannote_workflows.utils


class Baseline(sciluigi.WorkflowTask):

    workdir = luigi.Parameter(default='/work')

    series = luigi.Parameter(default='GameOfThrones')
    season = luigi.IntParameter(default=1)
    episode = luigi.IntParameter(default=1)
    language = luigi.Parameter(default='en')

    speech__fill_gaps = luigi.FloatParameter(default=1.0)

    bicSegmentationFeatures__e = luigi.BoolParameter(default=False)
    bicSegmentationFeatures__De = luigi.BoolParameter(default=False)
    bicSegmentationFeatures__DDe = luigi.BoolParameter(default=False)
    bicSegmentationFeatures__coefs = luigi.IntParameter(default=11)
    bicSegmentationFeatures__D = luigi.BoolParameter(default=False)
    bicSegmentationFeatures__DD = luigi.BoolParameter(default=False)

    bicSegmentation__penalty_coef = luigi.FloatParameter(default=1.0)
    bicSegmentation__covariance_type = luigi.Parameter(default='full')
    bicSegmentation__min_duration = luigi.FloatParameter(default=1.0)
    bicSegmentation__precision = luigi.FloatParameter(default=0.1)

    linearBICClusteringFeatures__e = luigi.BoolParameter(default=False)
    linearBICClusteringFeatures__De = luigi.BoolParameter(default=False)
    linearBICClusteringFeatures__DDe = luigi.BoolParameter(default=False)
    linearBICClusteringFeatures__coefs = luigi.IntParameter(default=11)
    linearBICClusteringFeatures__D = luigi.BoolParameter(default=False)
    linearBICClusteringFeatures__DD = luigi.BoolParameter(default=False)

    linearBICClustering__penalty_coef = luigi.FloatParameter(default=1.0)
    linearBICClustering__covariance_type = luigi.Parameter(default='diag')

    bicClusteringFeatures__e = luigi.BoolParameter(default=False)
    bicClusteringFeatures__De = luigi.BoolParameter(default=False)
    bicClusteringFeatures__DDe = luigi.BoolParameter(default=False)
    bicClusteringFeatures__coefs = luigi.IntParameter(default=11)
    bicClusteringFeatures__D = luigi.BoolParameter(default=False)
    bicClusteringFeatures__DD = luigi.BoolParameter(default=False)

    bicClustering__penalty_coef = luigi.FloatParameter(default=3.5)
    bicClustering__covariance_type = luigi.Parameter(default='full')

    hyperopt = luigi.Parameter(default=None)

    def workflow(self):

        # =====================================================================
        # SPEECH / NON-SPEECH
        # =====================================================================

        audio = self.new_task(
            'audio',
            pyannote_workflows.tasks.tvd_dataset.Audio,
            series=self.series,
            season=self.season,
            episode=self.episode,
            language=self.language)

        speakerReference = self.new_task(
            'speakerReference',
            pyannote_workflows.tasks.tvd_dataset.Speaker,
            workdir=self.workdir,
            series=self.series,
            season=self.season,
            episode=self.episode)

        speech = self.new_task(
            'speechReference',
            pyannote_workflows.tasks.tvd_dataset.Speech,
            fill_gaps=self.speech__fill_gaps)

        speech.in_wav = audio.out_put
        speech.in_speaker = speakerReference.out_put

        # =====================================================================
        # BIC SEGMENTATION
        # =====================================================================

        bicSegmentationFeatures = self.new_task(
            'bicSegmentationFeatures',
            pyannote_workflows.tasks.speech.MFCC,
            e=self.bicSegmentationFeatures__e,
            De=self.bicSegmentationFeatures__De,
            DDe=self.bicSegmentationFeatures__DDe,
            coefs=self.bicSegmentationFeatures__coefs,
            D=self.bicSegmentationFeatures__D,
            DD=self.bicSegmentationFeatures__DD)

        bicSegmentationFeatures.in_audio = audio.out_put

        bicSegmentation = self.new_task(
            'bicSegmentation',
            pyannote_workflows.tasks.speech.BICSegmentation,
            penalty_coef=self.bicSegmentation__penalty_coef,
            covariance_type=self.bicSegmentation__covariance_type,
            min_duration=self.bicSegmentation__min_duration,
            precision=self.bicSegmentation__precision,
        )

        bicSegmentation.in_segmentation = speech.out_put
        bicSegmentation.in_features = bicSegmentationFeatures.out_put

        # =====================================================================
        # LINEAR BIC CLUSTERING
        # =====================================================================

        linearBICClusteringFeatures = self.new_task(
            'linearBICClusteringFeatures',
            pyannote_workflows.tasks.speech.MFCC,
            e=self.linearBICClusteringFeatures__e,
            De=self.linearBICClusteringFeatures__De,
            DDe=self.linearBICClusteringFeatures__DDe,
            coefs=self.linearBICClusteringFeatures__coefs,
            D=self.linearBICClusteringFeatures__D,
            DD=self.linearBICClusteringFeatures__DD)

        linearBICClusteringFeatures.in_audio = audio.out_put

        linearBICClustering = self.new_task(
            'linearBICClustering',
            pyannote_workflows.tasks.speech.LinearBICClustering,
            penalty_coef=self.linearBICClustering__penalty_coef,
            covariance_type=self.linearBICClustering__covariance_type)

        linearBICClustering.in_segmentation = bicSegmentation.out_put
        linearBICClustering.in_features = linearBICClusteringFeatures.out_put

        # =====================================================================
        # BIC CLUSTERING
        # =====================================================================

        bicClusteringFeatures = self.new_task(
            'bicClusteringFeatures',
            pyannote_workflows.tasks.speech.MFCC,
            e=self.bicClusteringFeatures__e,
            De=self.bicClusteringFeatures__De,
            DDe=self.bicClusteringFeatures__DDe,
            coefs=self.bicClusteringFeatures__coefs,
            D=self.bicClusteringFeatures__D,
            DD=self.bicClusteringFeatures__DD)

        bicClusteringFeatures.in_audio = audio.out_put

        bicClustering = self.new_task(
            'bicClustering',
            pyannote_workflows.tasks.speech.BICClustering,
            penalty_coef=self.bicClustering__penalty_coef,
            covariance_type=self.bicClustering__covariance_type)

        bicClustering.in_segmentation = linearBICClustering.out_put
        bicClustering.in_features = bicClusteringFeatures.out_put

        # =====================================================================
        # EVALUATION
        # =====================================================================

        evaluateDiarization = self.new_task(
            'evaluateDiarization',
            pyannote_workflows.tasks.evaluation.EvaluateDiarizationFast)

        evaluateDiarization.in_hypothesis = bicClustering.out_put
        evaluateDiarization.in_reference = speakerReference.out_put

        if self.hyperopt is not None:
            hyperopt = self.new_task(
                'hyperopt',
                pyannote_workflows.utils.Hyperopt,
                temp=self.hyperopt)
            hyperopt.in_evaluation = evaluateDiarization.out_put
            return hyperopt

        else:
            return evaluateDiarization
