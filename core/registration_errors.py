"""Frame estimation failures shared by registration and its desktop consumers."""


class AlignmentEstimationError(ValueError):
    """The detector could not estimate a frame pair."""


class RejectedAlignmentFrame(AlignmentEstimationError):
    """An estimation failure associated with an input frame."""

    def __init__(self, method, frame_index, reason):
        super().__init__(method, frame_index, reason)
        self.method = method
        self.frame_index = frame_index
        self.reason = reason
        self.source_name = None

    def __str__(self):
        return f'{self.method} alignment failed at frame {self.frame_index + 1}: {self.reason}'
