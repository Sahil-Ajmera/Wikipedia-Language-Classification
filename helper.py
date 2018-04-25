class helper:
    __slots__ = 'attributes','seen','results','total_results', 'depth', 'prediction_at_this_stage','bool','value','left','right'

    def __init__(self,attributes,seen,results,total_results,depth,prediction_at_this_stage, boolean):
        self.attributes = attributes
        self.seen = seen
        self.results = results
        self.total_results = total_results
        self.depth = depth
        self.prediction_at_this_stage = prediction_at_this_stage
        self.bool = boolean
        self.value = None
        self.left = None
        self.right = None

