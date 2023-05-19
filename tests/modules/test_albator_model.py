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
      
    def test_ablate_model(self):
        # Create a model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10)
        )

        # Define an ablation strategy
        ablation_strategy = [
            "remove_layer",
            "remove_neurons"
        ]

        # Ablate the model
        ablated_model = ablate_model(model, ablation_strategy)

        # Check that the ablated model has the correct number of layers
        self.assertEqual(len(ablated_model), len(model) - 1)

        # Check that the ablated model has the correct number of neurons
        self.assertEqual(ablated_model[0].out_features, model[0].out_features - 1)

        # Check that the ablated model can still be used
        ablated_model(torch.rand(10))



def main():
    unittest.main()

if __name__ == "__main__":
    main()
