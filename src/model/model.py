class Model:
    def __init__(self, parameters):
        self.parameters = parameters

    def calculate(self):
        print(f"Calculating model for: {self.parameters}")