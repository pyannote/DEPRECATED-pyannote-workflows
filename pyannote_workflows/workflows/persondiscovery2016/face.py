import luigi
import sciluigi

import ssl
import cv2
import six.moves.urllib.request
import urllib
import pyannote.video
import pyannote.video.structure
import pyannote.video.face.face
import pyannote.video.face.tracking
import pyannote.video.face.clustering
from pyannote.core import Timeline
import pyannote.core.json

import pyannote_workflows.tasks.person_discovery_2016


class _ShotThreading(sciluigi.ExternalTask):

    workdir = luigi.Parameter()

    in_video = None

    def out_put(self):
        TEMPLATE = '{workdir}/_threads/{corpus}/{show}.json'
        video = self.in_video().task
        corpus = video.corpus
        show = video.show
        path = TEMPLATE.format(
            workdir=self.workdir, corpus=corpus, show=show)
        return sciluigi.TargetInfo(self, path)


class _DLIBModel(sciluigi.Task):

    workdir = luigi.Parameter()

    def out_put(self):
        TEMPLATE = '{workdir}/_models/dlib.face.landmarks.dat'
        path = TEMPLATE.format(workdir=self.workdir)
        return sciluigi.TargetInfo(self, path)

    def run(self):
        URL = "https://raw.githubusercontent.com/pyannote/pyannote-data/master/dlib.face.landmarks.dat"
        context = ssl._create_unverified_context()
        resource = six.moves.urllib.request.urlopen(URL, context=context)
        with self.out_put().open('wb') as fp:
            fp.write(resource.read())


class _OpenfaceModel(sciluigi.Task):

    workdir = luigi.Parameter()

    def out_put(self):
        TEMPLATE = '{workdir}/_models/openface.nn4.small2.v1.t7'
        path = TEMPLATE.format(workdir=self.workdir)
        return sciluigi.TargetInfo(self, path)

    def run(self):
        URL = "https://raw.githubusercontent.com/pyannote/pyannote-data/master/openface.nn4.small2.v1.t7"
        context = ssl._create_unverified_context()
        resource = six.moves.urllib.request.urlopen(URL, context=context)
        with self.out_put().open('wb') as fp:
            fp.write(resource.read())


class FaceTracking(sciluigi.Task):

    workdir = luigi.Parameter()
    in_video = None
    in_shot = None

    def out_put(self):
        TEMPLATE = '{workdir}/face_tracking/{corpus}/{show}.txt'
        video = self.in_video().task
        corpus = video.corpus
        show = video.show
        path = TEMPLATE.format(
            workdir=self.workdir, corpus=corpus, show=show)
        return sciluigi.TargetInfo(self, path)

    def run(self):

        FACE_TEMPLATE = ('{t:.3f} {identifier:d} '
                         '{left:.3f} {top:.3f} {right:.3f} {bottom:.3f}\n')

        video = pyannote.video.Video(self.in_video().path)

        with self.in_shot().open('r') as fp:
            shot = pyannote.core.json.load(fp)
        shot = shot.get_timeline()

        tracking = pyannote.video.face.tracking.FaceTracking(
            detect_min_size=0.1,
            detect_every=1.0,
            track_max_gap=1.0)

        with self.out_put().open('w') as fp:

            for identifier, track in enumerate(tracking(video, shot)):

                for t, (left, top, right, bottom), _ in track:

                    line = FACE_TEMPLATE.format(
                        t=t, identifier=identifier,
                        left=left, right=right, top=top, bottom=bottom)

                    fp.write(line)


class _FaceLandmarks(sciluigi.Task):

    workdir = luigi.Parameter()
    in_video = None
    in_tracking = None
    in_model = None

    def out_put(self):
        TEMPLATE = '{workdir}/_face_landmarks/{corpus}/{show}.txt'
        video = self.in_video().task
        corpus = video.corpus
        show = video.show
        path = TEMPLATE.format(
            workdir=self.workdir, corpus=corpus, show=show)
        return sciluigi.TargetInfo(self, path)

    def run(self):

        video = pyannote.video.Video(self.in_video().path)
        frame_width, frame_height = video.frame_size

        tracking = self.in_tracking().path
        face_generator = pyannote_workflows.task.person_discovery_2016._getFaceGenerator(
            tracking, frame_width, frame_height, double=False)
        face_generator.send(None)

        model = self.in_model().path
        face = pyannote.video.face.face.Face(landmarks=model)

        with self.out_put().open('w') as fp:

            for timestamp, rgb in video:

                # get all detected faces at this time
                T, faces = face_generator.send(timestamp)
                # not that T might be differ slightly from t
                # due to different steps in frame iteration

                for identifier, boundingBox, _ in faces:

                    landmarks = face._get_landmarks(rgb, boundingBox)

                    fp.write('{t:.3f} {identifier:d}'.format(
                        t=T, identifier=identifier))

                    for x, y in landmarks:
                        fp.write(' {x:.5f} {y:.5f}'.format(
                            x=x / frame_width,
                            y=y / frame_height))
                    fp.write('\n')


class _Openface(sciluigi.Task):

    workdir = luigi.Parameter()
    in_video = None
    in_landmarks = None
    in_model = None

    def out_put(self):
        TEMPLATE = '{workdir}/_openface/{corpus}/{show}.txt'
        video = self.in_video().task
        corpus = video.corpus
        show = video.show
        path = TEMPLATE.format(
            workdir=self.workdir, corpus=corpus, show=show)
        return sciluigi.TargetInfo(self, path)

    def run(self):

        video = pyannote.video.Video(self.in_video().path)
        frame_width, frame_height = video.frame_size

        landmarks = self.in_tracking().path
        landmark_generator = pyannote_workflows.task.person_discovery_2016._getLandmarkGenerator(
            landmarks, frame_width, frame_height, double=True)
        landmark_generator.send(None)

        model = self.in_model().path
        face = pyannote.video.face.face.Face(size=96, openface=model)

        with self.out_put().open('w') as fp:

            for timestamp, rgb in video:

                T, shapes = landmark_generator.send(timestamp)

                for identifier, landmarks in shapes:
                    normalized_rgb = face._get_normalized(rgb, landmarks)
                    normalized_bgr = cv2.cvtColor(normalized_rgb,
                                                  cv2.COLOR_BGR2RGB)
                    openface = face._get_openface(normalized_bgr)

                    fp.write('{t:.3f} {identifier:d}'.format(
                        t=T, identifier=identifier))
                    for x in openface:
                        fp.write(' {x:.5f}'.format(x=x))
                    fp.write('\n')


class FaceClustering(sciluigi.Task):

    workdir = luigi.Parameter()
    in_video = None
    in_openface = None

    def out_put(self):
        TEMPLATE = '{workdir}/face_clustering/{corpus}/{show}.txt'
        video = self.in_video().task
        corpus = video.corpus
        show = video.show
        path = TEMPLATE.format(
            workdir=self.workdir, corpus=corpus, show=show)
        return sciluigi.TargetInfo(self, path)

    def run(self):

        TEMPLATE = '{identifier:d} {cluster:g}\n'

        clustering = pyannote.video.face.clustering.FaceClustering(
            threshold=0.4)

        openface = self.in_openface().path
        starting_point, features = clustering.model.preprocess(openface)

        result = clustering(starting_point, features=features)

        with self.out_put().open('w') as fp:
            for _, identifier, cluster in result.itertracks(label=True):
                line = TEMPLATE.format(identifier=identifier, cluster=cluster)
                fp.write(line)


class FaceWorkflow(sciluigi.WorkflowTask):

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

        _shotThreading = self.new_task(
            '_shotThreading',
            _ShotThreading,
            workdir=self.workdir)

        _shotThreading.in_video = video.out_put

        faceTracking = self.new_task(
            'faceTracking',
            FaceTracking,
            workdir=self.workdir)

        faceTracking.in_video = video.out_put
        faceTracking.in_shot = _shotThreading.out_put

        _faceLandmarks = self.new_task(
            '_faceLandmarks',
            _FaceLandmarks,
            workdir=self.workdir)

        _dlibModel = self.new_task(
            '_dlibModel',
            _DLIBModel,
            workdir=self.workdir)

        _faceLandmarks.in_video = video.out_put
        _faceLandmarks.in_tracking = faceTracking.out_put
        _faceLandmarks.in_model = _dlibModel.out_put

        _openfaceModel = self.new_task(
            '_openfaceModel',
            _OpenfaceModel,
            workdir=self.workdir)

        _openface = self.new_task(
            '_openface',
            _Openface,
            workdir=self.workdir)

        _openface.in_video = video.out_put
        _openface.in_landmarks = _faceLandmarks.out_put
        _openface.in_model = _openfaceModel.out_put

        faceClustering = self.new_task(
            'faceClustering',
            FaceClustering,
            workdir=self.workdir)

        faceClustering.in_video = video.out_put
        faceClustering.in_openface = _openface.out_put

        return faceClustering


if __name__ == '__main__':
        sciluigi.run_local(main_task_cls=FaceWorkflow)
