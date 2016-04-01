import luigi
import sciluigi
import pyannote.algorithms.tagging
import pyannote.core.json

from pyannote_workflows.utils import AutoOutput


class ConservativeDirectTagging(sciluigi.Task, AutoOutput):

    in_source = None
    in_target = None

    def run(self):

        # source (with string labels)
        with self.in_source().open('r') as fp:
            source = pyannote.core.json.load(fp)

        # target (with integer labels)
        with self.in_target().open('r') as fp:
            target = pyannote.core.json.load(fp)
            target = target.anonymize_labels(generator='int')

        tagging = pyannote.algorithms.tagging.ConservativeDirectTagger()
        tagged = tagging(source, target)

        tagged = tagged.anonymize_labels(generator='string')

        with self.out_put().open('w') as fp:
            pyannote.core.json.dump(tagged, fp)


class ArgMaxTagging(sciluigi.Task, AutoOutput):

    in_source = None
    in_target = None

    def run(self):

        # source (with string labels)
        with self.in_source().open('r') as fp:
            source = pyannote.core.json.load(fp)

        # target (with integer labels)
        with self.in_target().open('r') as fp:
            target = pyannote.core.json.load(fp)
            target = target.anonymize_labels(generator='int')

        tagging = pyannote.algorithms.tagging.ArgMaxTagger()
        tagged = tagging(source, target)

        tagged = tagged.anonymize_labels(generator='string')

        with self.out_put().open('w') as fp:
            pyannote.core.json.dump(tagged, fp)
