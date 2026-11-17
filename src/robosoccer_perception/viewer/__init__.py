"""Interactive debugging viewer for the vision-localization pipeline. See docs/viewer.md.

    session.py     one scenario run frame by frame: simulator on one side, the Phase 3
                   pipeline on the other, one `ViewerFrame` per step
    frame.py       the `ViewerFrame` packet every panel is drawn from, and its packed form
    recording.py   recorded runs on disk (one ``.npz``), read lazily for scrubbing
    render.py      the four panels as RGB arrays (NumPy / OpenCV; no GUI toolkit)
    worker.py      the thread that runs the session so the window never waits for inference
    headless.py    record a run, export a video or a contact sheet without a window
    app.py         the PySide6 window

Only `app.py` and `qt_env.py` need PySide6. The viewer is an observer: it shows ground truth
next to the estimate, and `session.py` keeps that ground truth away from the estimator.
"""
