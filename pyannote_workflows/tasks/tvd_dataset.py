import luigi
import sciluigi

import contextlib
import wave

from tvd import series_plugins
from tvd import Episode

import pyannote.core.json
from pyannote.core import Segment, Timeline, Annotation
import pyannote.parser

from pyannote_workflows.utils import AutoOutput


def get_episode(task):
    return Episode(series=task.series,
                   season=task.season,
                   episode=task.season)


class Audio(sciluigi.ExternalTask):

    series = luigi.Parameter()
    season = luigi.IntParameter()
    episode = luigi.IntParameter()
    language = luigi.Parameter(default=None)

    def out_put(self):

        dataset = series_plugins[self.series]('/tvd', acknowledgment=False)
        episode = get_episode(self)
        path = dataset.path_to_audio(episode, language=self.language)
        return sciluigi.TargetInfo(self, path)


class Subtitles(sciluigi.Task, AutoOutput):

    series = luigi.Parameter()
    season = luigi.IntParameter()
    episode = luigi.IntParameter()
    language = luigi.Parameter(default=None)

    def run(self):

        dataset = series_plugins[self.series]('/tvd', acknowledgment=False)
        episode = get_episode(self)
        path = dataset.path_to_subtitles(episode, language=self.language)

        parser = pyannote.parser.SRTParser(split=True, duration=True)
        transcription = parser.read(path)()

        with self.out_put().open('w') as fp:
            pyannote.core.json.dump(transcription, fp)


class SubtitlesTimeline(sciluigi.Task, AutoOutput):
    """Timeline containing one segment per subtitle timespans"""

    in_subtitles = None

    def run(self):
        with self.in_subtitles().open('r') as fp:
            transcription = pyannote.core.json.load(fp)
        timeline = Timeline()
        for start, end, edge in transcription.ordered_edges_iter(data=True):
            if 'subtitle' not in edge:
                continue
            segment = Segment(start, end)
            timeline.add(segment)

        with self.out_put().open('w') as fp:
            pyannote.core.json.dump(timeline, fp)


class SubtitlesAnnotation(sciluigi.Task, AutoOutput):
    """Annotation containing one label per subtitle timespans"""

    in_subtitles = None

    def run(self):
        with self.in_subtitles().open('r') as fp:
            transcription = pyannote.core.json.load(fp)
        annotation = Annotation()
        label = 0
        for start, end, edge in transcription.ordered_edges_iter(data=True):
            if 'subtitle' not in edge:
                continue
            segment = Segment(start, end)
            annotation[segment] = label
            label += 1

        with self.out_put().open('w') as fp:
            pyannote.core.json.dump(annotation, fp)



class SubtitlesSpeechNonSpeech(sciluigi.Task, AutoOutput):
    """Annotation containing 'speech' and 'non-speech' labels based
    on subtitles timespans"""

    in_wav = None
    in_subtitles = None

    def run(self):

        # wav file duration
        wav = self.in_wav().path
        with contextlib.closing(wave.open(wav, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
        duration = frames / rate
        extent = Segment(0., duration)

        with self.in_subtitles().open('r') as fp:
            transcription = pyannote.core.json.load(fp)
        annotation = Annotation()
        for start, end, edge in transcription.ordered_edges_iter(data=True):
            if 'subtitle' not in edge:
                continue
            segment = Segment(start, end)
            annotation[segment] = 'speech'

        for gap in annotation.get_timeline().gaps(extent):
            annotation[gap] = 'non_speech'

        with self.out_put().open('w') as fp:
            pyannote.core.json.dump(annotation, fp)


class Speaker(sciluigi.Task, AutoOutput):

    series = luigi.Parameter()
    season = luigi.IntParameter()
    episode = luigi.IntParameter()

    def run(self):
        dataset = series_plugins[self.series]('/tvd', acknowledgment=False)
        episode = get_episode(self)

        speaker = dataset.get_resource('speaker', episode)

        with self.out_put().open('w') as fp:
            pyannote.core.json.dump(speaker, fp)


class SpeechNonSpeech(sciluigi.Task, AutoOutput):

    in_wav = None
    in_speaker = None

    def run(self):

        # wav file duration
        wav = self.in_wav().path
        with contextlib.closing(wave.open(wav, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
        duration = frames / rate
        extent = Segment(0., duration)

        with self.in_speaker().open('r') as fp:
            speaker = pyannote.core.json.load(fp)

        segmentation = Annotation()
        for segment, _ in speaker.itertracks():
            segmentation[segment] = 'speech'
        segmentation = segmentation.smooth()

        for gap in segmentation.get_timeline().gaps(extent):
                segmentation[gap] = 'non_speech'
        segmentation = segmentation.smooth()

        with self.out_put().open('w') as fp:
            pyannote.core.json.dump(segmentation, fp)


class Speech(sciluigi.Task, AutoOutput):

    fill_gaps = luigi.FloatParameter(default=1.000)

    in_wav = None
    in_speaker = None

    def run(self):

        # wav file duration
        wav = self.in_wav().path
        with contextlib.closing(wave.open(wav, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
        duration = frames / rate
        extent = Segment(0., duration)

        with self.in_speaker().open('r') as fp:
            speaker = pyannote.core.json.load(fp)

        timeline = Timeline()
        for segment, _ in speaker.itertracks():
            timeline.add(segment)

        # fill gaps
        for gap in timeline.gaps(extent):
            if gap.duration < self.fill_gaps:
                timeline.add(gap)
        timeline = timeline.coverage()

        with self.out_put().open('w') as fp:
            pyannote.core.json.dump(timeline, fp)
