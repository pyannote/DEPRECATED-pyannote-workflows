import luigi
import sciluigi

import pyannote.video
import pyannote.video.structure
from pyannote.core import Timeline
import pyannote.core.json
import pyannote_workflows.tasks.person_discovery_2016


class _ShotBoundaryDetection(sciluigi.Task):

    workdir = luigi.Parameter()

    in_video = None

    def out_put(self):
        TEMPLATE = '{workdir}/_shots/{corpus}/{show}.json'
        video = self.in_video().task
        corpus = video.corpus
        show = video.show
        path = TEMPLATE.format(
            workdir=self.workdir, corpus=self.corpus, show=self.show)
        return sciluigi.TargetInfo(self, path)

    def run(self):
        video = pyannote.video.Video(self.in_video().path)
        shots = Timeline(pyannote.video.structure.Shot(video))

        with self.out_put().open('w') as fp:
            pyannote.core.json.dump(shots, fp)


class _ShotThreading(sciluigi.Task):

    workdir = luigi.Parameter()

    in_video = None
    in_shot = None

    def out_put(self):
        TEMPLATE = '{workdir}/_threads/{corpus}/{show}.json'
        video = self.in_video().task
        corpus = video.corpus
        show = video.show
        path = TEMPLATE.format(
            workdir=self.workdir, corpus=self.corpus, show=self.show)
        return sciluigi.TargetInfo(self, path)

    def run(self):
        video = pyannote.video.Video(self.in_video().path)

        with self.in_shot().open('r') as fp:
            shot = pyannote.core.json.load(fp)

        threads = pyannote.video.structure.Thread(video, shot=shot)
        threads = threads.smooth()

        with self.out_put().open('w') as fp:
            pyannote.core.json.dump(threads, fp)


class BaselineShots(sciluigi.Task):

    workdir = luigi.Parameter()

    in_video = None
    in_thread = None

    def out_put(self):
        TEMPLATE = '{workdir}/shots/{corpus}/{show}.shot'
        video = self.in_video().task
        corpus = video.corpus
        show = video.show
        path = TEMPLATE.format(
            workdir=self.workdir, corpus=self.corpus, show=self.show)
        return sciluigi.TargetInfo(self, path)

    def run(self):

        # INA F2_TS/20130607/130607FR20000_B.MPG 000123 00595.400 00598.240
        TEMPLATE = '{corpus} {show} {index:06d} {start:09.3f} {end:09.3f}\n'

        with self.in_thread().open('r') as fp:
            threads = pyannote.core.json.load(fp)

        with self.out_put().open('w') as fp:
            for s, (segment, _) in enumerate(threads.itertracks()):
                line = TEMPLATE.format(
                    corpus=corpus, show=show, index=s,
                    start=segment.start, end=segment.end)
                fp.write(line)


class ShotWorkflow(sciluigi.WorkflowTask):

    workdir = luigi.Parameter(
        default='/vol/work1/bredin/mediaeval/PersonDiscovery2016/baseline')

    corpus_dir = luigi.Parameter(
        default='/vol/corpora5/mediaeval')

    corpus = luigi.Parameter(
        default='INA')

    show = luigi.Parameter(
        default='F2_TS/20130607/130607FR20000_B.MPG')

    def workflow(self):

        video = self.new_task(
            'video',
            pyannote_workflows.tasks.person_discovery_2016.Video,
            corpus_dir=self.corpus_dir,
            corpus=self.corpus,
            show=self.show)

        _shotBoundaryDetection = self.new_task(
            '_shotBoundaryDetection',
            _ShotBoundaryDetection,
            workdir=self.workdir)

        _shotBoundaryDetection.in_video = video.out_put

        _shotThreading = self.new_task(
            '_shotThreading',
            _ShotThreading,
            workdir=self.workdir)

        _shotThreading.in_video = video.out_put
        _shotThreading.in_shot = _shotBoundaryDetection.out_put

        baselineShots = self.new_task(
            'baselineShots',
            BaselineShots,
            workdir=self.workdir)

        baselineShots.in_video = video.out_put
        baselineShots.in_thread = _shotThreading.out_put

        return baselineShots
