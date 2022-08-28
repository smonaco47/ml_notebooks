import time
import numpy as np

import torch
from sklearn.metrics import confusion_matrix


# TODO pre-commit black, mypy, etc

class ModelTrainer:
    def __init__(self, device, n_labels, n_epochs = 10):
        self.print_interval = 50
        self.shortcut_quit = 1000000
        self.max_train_seconds = 60 * 60 * 5
        self.start_time = time.time()
        self.n_labels = n_labels
        self.n_epochs = n_epochs
        self.min_increase_threshold = 0.05
        self.device = device

        self.num_workers = 2
        self.batch_size = 100

    def train_model(self, model, criterion, optimizer, train_loader, test_loader):
        loss_history = []
        validation_history = []
        training_history = []
        final_accuracy = 0
        final_accuracy_by_label = None
        final_conf_matrix = None
        minimal_increase = 0
        for epoch in range(self.n_epochs):
            print(f"starting epoch {epoch+1} of {self.n_epochs}")
            loss_history.append(self._train_single_epoch(model, train_loader, self.device, optimizer, criterion))


        #     Calculate model accuracy on training samples
            outputs, targets = self._validate_model(model, train_loader, self.device)
            accuracy, _, _ = self._calculate_prediction_accuracy_by_label(outputs, targets)
            training_history.append(accuracy)
            print(f"..\ttrain\t{accuracy}")

        #     Calculate model accuracy on validation samples
            outputs, targets = self._validate_model(model, test_loader, self.device)
            validation_accuracy, final_accuracy_by_label, final_conf_matrix = self._calculate_prediction_accuracy_by_label(outputs, targets)
            validation_history.append(validation_accuracy)

            if self.min_increase_threshold < (final_accuracy - validation_accuracy)/validation_accuracy:
                minimal_increase += 1
            else:
                minimal_increase = 0

            final_accuracy = validation_accuracy
            print(f"..\tval\t{final_accuracy}")

            if minimal_increase == 3:
                print("Validation is not improving, quitting")
                break


            if time.time() - self.start_time > self.max_train_seconds:
                print("Maximum time limit hit, proceeding with validation.")
                break
        return {
            "final_accuracy": final_accuracy, 
            "final_accuracy_by_label": final_accuracy_by_label, 
            "loss_history": loss_history,
            "validation_history": validation_history,
            "training_history": training_history,
            "final_conf_matrix": final_conf_matrix,
            "n_epochs": epoch,
            "model": model
        }

    def _train_single_epoch(self, mod, loader, device, optim, crit):
        mod.train()
        losses = []
        for i, (images, targets) in enumerate(loader):
            images, targets = images.to(device), targets.to(device)
            output = mod(images)
            loss = crit(output, targets)

            optim.zero_grad()
            loss.backward()
            optim.step()
            
            losses.append(float(loss))

            if (i % self.print_interval) == 0: 
                print(f"... \tloss_{i}\t{float(loss)}")
            if i > self.shortcut_quit:
                break
            if time.time() - self.start_time > self.max_train_seconds:
                break
        print(f"..\tavg_l\t{sum(losses) / len(losses)}")  
        return losses


    def _validate_model(self, mod, loader, device):
        mod.eval()
        predicted_values = np.empty((0, self.n_labels), float)
        expected_targets = np.empty((0,1), int)
        with torch.no_grad():
            for idx, (images, target) in enumerate(loader):
                images, target = images.to(device), target.to(device)

                output = mod(images)
                predicted_values =  np.append(predicted_values, output.cpu().numpy(), axis=0)
                expected_targets = np.append(expected_targets, target.cpu())
                if idx > self.shortcut_quit:
                    break
                    
        return predicted_values, expected_targets


    def _calculate_prediction_accuracy_by_label(self, predicted_prob, expected_labels):
        predicted_targets = np.argmax(predicted_prob, axis=1)
        matrix = confusion_matrix(expected_labels, predicted_targets)
        total = matrix.sum()
        correct = np.identity(self.n_labels) * matrix
        accuracy_by_label = correct.sum(axis=1) / matrix.sum(axis=1)
        return correct.sum()/total, accuracy_by_label, matrix,
