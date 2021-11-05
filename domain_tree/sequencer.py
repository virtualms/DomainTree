class Singleton(object):
    _instance = None

    def __new__(class_, *args, **kwargs):
        if not isinstance(class_._instance, class_):
            class_._instance = object.__new__(class_, *args, **kwargs)
        return class_._instance


# dummy sequencer for node number
class Sequencer(Singleton):
    sequencer = -1

    def get_seq_num(self):
        self.sequencer = self.sequencer + 1
        return self.sequencer

    def reset(self):
        self.sequencer = -1

    def __exit__(self):
        self.reset()

    def __enter__(self):
        self.reset()
