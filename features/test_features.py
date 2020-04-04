import time


def cal():
    """A test calculation of some indicators. """

    print('%s | A test feature is being calculated.' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


if __name__ == '__main__':
    cal()