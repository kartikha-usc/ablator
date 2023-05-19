import unittest

from ablator import ablate_model

class AblateModelTest(unittest.TestCase):

    def test_ablate_model_with_invalid_strategy(self):
        # Create a model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10)
        )

        # Define an invalid ablation strategy
        ablation_strategy = "remove_layer_that_doesnt_exist"

        # Try to ablate the model
        with self.assertRaises(ValueError):
            ablate_model(model, ablation_strategy)

def main():
    unittest.main()

if __name__ == "__main__":
    main()
