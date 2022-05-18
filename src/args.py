class Args:
    def __init__(self):
        self.output_dir = None
        self.data_dir = None
        self.data_filepath = None
        self.base_model = None
        self.batch_size = 2
        self.weight_decay = (0.01,)
        self.logging_dir = self.output_dir
        self.prediction_loss_only = True
        self.epochs = 1
        self.text_column = None
        self.group_column = None
        self.filter_by = None


args = Args()
