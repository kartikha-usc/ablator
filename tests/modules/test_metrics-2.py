import unittest

from ablator import metrics

class MetricsTest(unittest.TestCase):

    def test_accuracy_with_more_data(self):
        # Create a model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10)
        )

        # Create more data
        data = [
            (torch.rand(10), torch.randint(0, 10, (10,))),
            (torch.rand(10), torch.randint(0, 10, (10,))),
            (torch.rand(10), torch.randint(0, 10, (10,))),
            (torch.rand(10), torch.randint(0, 10, (10,))),
            (torch.rand(10), torch.randint(0, 10, (10,))),
            (torch.rand(10), torch.randint(0, 10, (10,)))
        ]

        # Train the model
        model.train()
        for x, y in data:
            model.zero_grad()
            y_hat = model(x)
            loss = metrics.cross_entropy(y_hat, y)
            loss.backward()
            optimizer.step()

        # Test the model
        model.eval()
        correct = 0
        total = 0
        for x, y in data:
            y_hat = model(x)
            _, predicted = y_hat.max(1)
            correct += (predicted == y).sum().item()
            total += len(y)

        # Calculate the accuracy
        accuracy = correct / total

        # Check that the accuracy is correct
        self.assertAlmostEqual(accuracy, 0.75)

    def test_accuracy_with_variety_of_data(self):
        # Create a model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10)
        )

        # Create a variety of data
        data = [
            (torch.rand(10), torch.randint(0, 10, (10,))),
            (torch.rand(10), torch.randint(0, 10, (10,))),
            (torch.rand(10), torch.randint(0, 10, (10,))),
            (torch.rand(10), torch.randint(0, 10, (10,))),
            (torch.rand(10), torch.randint(0, 10, (10,))),
            (torch.rand(10), torch.randint(0, 10, (10,)))
        ]

        # Train the model
        model.train()
        for x, y in data:
            model.zero_grad()
            y_hat = model(x)
            loss = metrics.cross_entropy(y_hat, y)
            loss.backward()
            optimizer.step()

        # Test the model
        model.eval()
        correct = 0
        total = 0
        for x, y in data:
            y_hat = model(x)
            _, predicted = y_hat.max(1)
            correct += (predicted == y).sum().item()
            total += len(y)

        # Calculate the accuracy
        accuracy = correct / total

        # Check that the accuracy is correct
        self.assertAlmostEqual(accuracy, 0.75)
        

    def test_accuracy_with_complex_models(self):
        # Create a complex model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 10)
        )
    
        # Create some data
        data = [
            (torch.rand(10), torch.randint(0, 10, (10,))),
            (torch.rand(10), torch.randint(0, 10, (10,))),
            (torch.rand(10), torch.randint(0, 10, (10,)))
        ]
    
        # Train the model
        model.train()
        for x, y in data:
            model.zero_grad()
            y_hat = model(x)
            loss = metrics.cross_entropy(y_hat, y)
            loss.backward()
            optimizer.step()
    
        # Test the model
        model.eval()
        correct = 0
        total = 0
        for x, y in data:
            y_hat = model(x)
            _, predicted = y_hat.max(1)
            correct += (predicted == y).sum().item()
            total += len(y)
    
        # Calculate the accuracy
        accuracy = correct / total
    
        # Check that the accuracy is correct
        self.assertAlmostEqual(accuracy, 0.75)
    
        # Create a more complex model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 10)
        )
    
        # Train the more complex model
        model.train()
        for x, y in data:
            model.zero_grad()
            y_hat = model(x)
            loss = metrics.cross_entropy(y_hat, y)
            loss.backward()
            optimizer.step()
    
        # Test the more complex model
        model.eval()
        correct = 0
        total = 0
        for x, y in data:
            y_hat = model(x)
            _, predicted = y_hat.max(1)
            correct += (predicted == y).sum().item()
            total += len(y)
    
        # Calculate the accuracy of the more complex model
        accuracy = correct / total
    
        # Check that the accuracy of the more complex model is higher than the accuracy of the simpler model
        self.assertGreater(accuracy, 0.75)

        
    def test_accuracy(self):
        # Create a model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10)
        )

        # Create some data
        data = [
            (torch.rand(10), torch.randint(0, 10, (10,))),
            (torch.rand(10), torch.randint(0, 10, (10,))),
            (torch.rand(10), torch.randint(0, 10, (10,)))
        ]

        # Train the model
        model.train()
        for x, y in data:
            model.zero_grad()
            y_hat = model(x)
            loss = metrics.cross_entropy(y_hat, y)
            loss.backward()
            optimizer.step()

        # Test the model
        model.eval()
        correct = 0
        total = 0
        for x, y in data:
            y_hat = model(x)
            _, predicted = y_hat.max(1)
            correct += (predicted == y).sum().item()
            total += len(y)

        # Calculate the accuracy
        accuracy = correct / total

        # Check that the accuracy is correct
        self.assertAlmostEqual(accuracy, 0.75)
 

def main():
    unittest.main()

if __name__ == "__main__":
    main()

