from tensorboardX import SummaryWriter

from ..utils.singleton import Singleton


@Singleton
class Logger(object):

    def __init__(self):
        self.summary_writer = None

    def setup(self, *args, **kwargs):
        if self.summary_writer is not None:
            raise RuntimeError("set_up can only be called once")
        self.summary_writer = SummaryWriter(*args, **kwargs)

    def add_audio(self, *args, **kwargs):
        self.ensure_ready()
        self.summary_writer.add_audio(*args, **kwargs)

    def add_embedding(self, *args, **kwargs):
        self.ensure_ready()
        self.summary_writer.add_embedding(*args, **kwargs)

    def ensure_ready(self):
        if self.summary_writer is None:
            raise RuntimeError("set_up has not been run")
