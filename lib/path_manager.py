from pathlib import Path

class PathManager():
    def __init__(self) -> None:
        cur_path = Path(__file__)

        self.root = cur_path.parent.parent
        assert (self.root.exists()), "Something unexpected happened. Root path does not exist."

        self.inputs  = self.root / 'data_inputs'
        self.outputs = self.root / 'data_outputs'
        assert (self.inputs.exists()), "Make sure you have a \'data_inputs\' folder in the root directory."
        assert (self.outputs.exists()), "Make sure you have a \'data_outputs\' folder in the root directory."
