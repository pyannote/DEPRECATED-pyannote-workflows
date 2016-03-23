import luigi
import sciluigi
import pyannote.metrics.detection
import pyannote.metrics.diarization
import pyannote.core.json

from pyannote_workflows.utils import AutoOutput


class EvaluateSpeechActivityDetection(sciluigi.Task, AutoOutput):

    in_reference = None
    in_hypothesis = None

    def run(self):

        with self.in_reference().open('r') as fp:
            reference = pyannote.core.json.load(fp)

        with self.in_hypothesis().open('r') as fp:
            hypothesis = pyannote.core.json.load(fp)

        detectionErrorRate = pyannote.metrics.detection.DetectionErrorRate()
        detection = detectionErrorRate(
            reference.subset(['speech']),
            hypothesis.subset(['speech']),
            detailed=True)

        results = {
            'reference': reference,
            'hypothesis': hypothesis,
            'evaluation': {
                'detection': detection
            }
        }

        with self.out_put().open('w') as fp:
            pyannote.core.json.dump(results, fp)


class EvaluateDiarization(sciluigi.Task, AutoOutput):

    in_hypothesis = None
    in_reference = None

    def run(self):

        with self.in_reference().open('r') as f:
            reference = pyannote.core.json.load(f)

        with self.in_hypothesis().open('r') as f:
            hypothesis = pyannote.core.json.load(f)

        purity = pyannote.metrics.diarization.DiarizationPurity()
        coverage = pyannote.metrics.diarization.DiarizationCoverage()
        diarization = pyannote.metrics.diarization.DiarizationErrorRate()

        results = {
            'reference': reference,
            'hypothesis': hypothesis,
            'evaluation': {
                'purity': purity(reference, hypothesis, detailed=True),
                'coverage': coverage(reference, hypothesis, detailed=True),
                'diarization': diarization(reference, hypothesis, detailed=True)
            }
        }

        with self.out_put().open('w') as fp:
            pyannote.core.json.dump(results, fp)
