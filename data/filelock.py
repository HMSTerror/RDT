import os

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None
    import msvcrt


class FileLock:
    """
    A file lock class.
    """
    def __init__(self, filename):
        self.filename = filename
        self.handle = None

    def acquire_read_lock(self):
        lock_path = self.filename + '.lock'
        if fcntl is not None:
            # Create lock file if needed, then reopen for shared locking.
            open(lock_path, "a").close()
            self.handle = open(lock_path, "r")
            fcntl.flock(self.handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        else:
            # Windows fallback: use a byte-range lock on the first byte.
            self.handle = open(lock_path, "a+b")
            self._win_lock()

    def acquire_write_lock(self):
        lock_path = self.filename + '.lock'
        if fcntl is not None:
            self.handle = open(lock_path, "w")
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            self.handle = open(lock_path, "a+b")
            self._win_lock()

    def _win_lock(self):
        # Ensure there is at least one byte to lock.
        self.handle.seek(0, os.SEEK_END)
        if self.handle.tell() == 0:
            self.handle.write(b"\0")
            self.handle.flush()
        self.handle.seek(0)
        msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)

    def release_lock(self):
        if self.handle is not None:
            if fcntl is not None:
                fcntl.flock(self.handle, fcntl.LOCK_UN)
            else:
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            self.handle.close()
            self.handle = None
