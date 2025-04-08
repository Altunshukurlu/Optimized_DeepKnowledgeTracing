import os
import time
import math
import random
import glob
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Constants
OUTPUT_ROOT = "output/synthetic/"
DATA_ROOT = "data/synthetic/"
N_QUESTIONS = 50

#--------------------------
# Data Loading and Processing
#--------------------------
class Data:
    """Data handler for loading all synthetic data files"""
    def __init__(self, n_questions=N_QUESTIONS):
        self.n_questions = n_questions
        
        print("Loading all CSV files from synthetic dataset directory")
        self.train_data, self.test_data = self._load_all_files()
            
        # Set student counts based on loaded data
        self.n_students = len(self.train_data)
        self.n_steps = self.n_questions - 1
        self.nTest = len(self.test_data)
        self.nTrain = len(self.train_data)
        
        print(f"Loaded {self.nTrain} training samples and {self.nTest} test samples")
    
    def _load_all_files(self):
        """Load and combine data from all CSV files in the synthetic directory"""
        all_train_data = []
        all_test_data = []
        
        # Find all CSV files
        csv_files = glob.glob(os.path.join(DATA_ROOT, "*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {DATA_ROOT}")
        
        print(f"Found {len(csv_files)} CSV files")
        
        for file_path in csv_files:
            print(f"Processing {os.path.basename(file_path)}")
            try:
                # Load the data
                raw_data = pd.read_csv(file_path, header=None).values
                
                # Check if file has expected format
                file_n_questions = raw_data.shape[1]
                if file_n_questions != self.n_questions:
                    print(f"Skipping {file_path} - has {file_n_questions} questions instead of {self.n_questions}")
                    continue
                
                # Split into train and test sets for this file
                total_students = raw_data.shape[0]
                n_students = total_students // 2
                
                train_tensor = torch.tensor(raw_data[:n_students], dtype=torch.float)
                test_tensor = torch.tensor(raw_data[n_students:total_students], dtype=torch.float)
                
                # Process and add to our collections
                train_processed = self._compress_data(train_tensor)
                test_processed = self._compress_data(test_tensor)
                
                all_train_data.extend(train_processed)
                all_test_data.extend(test_processed)
                
                print(f"  Added {len(train_processed)} training and {len(test_processed)} test examples")
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        if not all_train_data or not all_test_data:
            raise ValueError("No valid data files could be processed")
        
        print(f"Successfully loaded {len(all_train_data)} training examples and {len(all_test_data)} test examples")
        return all_train_data, all_test_data
    
    def _compress_data(self, dataset):
        """Convert tensor dataset to list of dictionaries format"""
        new_dataset = []
        for i in range(dataset.shape[0]):
            answers = self._compress_answers(dataset[i])
            new_dataset.append(answers)
        return new_dataset
    
    def _compress_answers(self, answers):
        """Convert a single student's answers to the required format"""
        new_answers = {
            'question_id': torch.arange(1, self.n_questions + 1),  # 1-indexed to match Lua
            'time': torch.zeros(self.n_questions),
            'correct': answers.clone(),
            'n_answers': self.n_questions
        }
        return new_answers
    
    def get_test_data(self):
        """Return the test dataset"""
        return self.test_data
    
    def get_train_data(self):
        """Return the train dataset"""
        return self.train_data

#--------------------------
# RNN Model
#--------------------------
class RNNLayer(nn.Module):
    def __init__(self, n_input, n_hidden, n_questions, dropout_prob=0.5):
        super(RNNLayer, self).__init__()
        
        # Memory and input projections
        self.lin_m = nn.Linear(n_hidden, n_hidden)
        self.lin_x = nn.Linear(n_input, n_hidden)
        self.lin_y = nn.Linear(n_hidden, n_questions)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)
        self.use_dropout = dropout_prob > 0
        
        # Define activation functions
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        # Loss function
        self.criterion = nn.BCELoss(reduction='none')
    
    def forward(self, memory, input_x, input_y, truth=None):
        # Process memory and inputs
        hidden = self.tanh(self.lin_m(memory) + self.lin_x(input_x))
        
        # Apply dropout if enabled and in training mode
        pred_input = self.dropout(hidden) if self.use_dropout and self.training else hidden
        
        # Compute predictions
        pred_output = self.sigmoid(self.lin_y(pred_input))
        pred = torch.sum(pred_output * input_y, dim=1)
        
        # Calculate loss if truth is provided
        loss = self.criterion(pred, truth) if truth is not None else None
        
        return pred, loss, hidden


class RNN(nn.Module):
    def __init__(self, params):
        super(RNN, self).__init__()
        
        # Store configuration parameters
        self.n_questions = params['n_questions']
        self.n_hidden = params['n_hidden']
        self.use_dropout = params.get('dropout', False)
        self.max_grad = params.get('max_grad', 10)
        self.max_steps = params.get('max_steps', 100)
        self.dropout_pred = params.get('dropout_pred', False)
        
        # Input dimensions
        self.n_input = self.n_questions * 2
        
        # Compressed sensing support
        self.compressed_sensing = params.get('compressed_sensing', False)
        if self.compressed_sensing:
            self.n_input = params['compressed_dim']
            torch.manual_seed(12345)
            self.register_buffer('basis', torch.randn(self.n_questions * 2, self.n_input))
        
        # Create model components
        self.start = nn.Linear(1, self.n_hidden)
        self.layer = RNNLayer(
            self.n_input, 
            self.n_hidden, 
            self.n_questions, 
            dropout_prob=0.5 if self.use_dropout else 0
        )
        
        # Create optimizer
        rate = 0.001
        self.optimizer = optim.AdamW(self.parameters(), lr=rate)
        print('RNN initialized')
    
    def calc_grad(self, batch, rate, alpha):
        """Forward and backward pass with gradient scaling"""
        # Get batch dimensions
        n_steps = get_n_steps(batch)
        n_students = len(batch)
        
        # Initialize tracking variables
        sum_err = 0
        num_tests = 0
        
        # Create initial state
        zero_input = torch.zeros(n_students, 1)
        state = self.start(zero_input)
        
        # Forward pass
        for k in range(1, n_steps + 1):
            input_x, input_y, truth = self.get_inputs(batch, k)
            mask = get_mask(batch, k)
            
            # Pass through the layer
            pred, loss, state = self.layer(state, input_x, input_y, truth)
            
            # Scale the loss
            scaled_loss = loss * alpha
            scaled_loss.mean().backward(retain_graph=True)
            
            # Update metrics
            step_err = loss.sum().item()
            num_tests += mask.sum().item()
            sum_err += step_err
        
        # Clip gradients to prevent explosion
        max_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad)
        
        return sum_err, num_tests, max_norm
    
    def get_inputs(self, batch, k):
        n_students = len(batch)
        
        input_x = torch.zeros(n_students, 2 * self.n_questions)
        input_y = torch.zeros(n_students, self.n_questions)
        truth = torch.zeros(n_students)
        
        for i, answers in enumerate(batch):
            if k >= 0:
                k-=1
            if k + 1 <= answers['n_answers']:
                current_id = answers['question_id'][k]
                next_id = answers['question_id'][k + 1]
                current_correct = answers['correct'][k]
                next_correct = answers['correct'][k + 1]
                
                # Set one-hot encoding for current question
                x_index = int(current_correct * self.n_questions + current_id)
                input_x[i, x_index - 1] = 1  # -1 because Python is 0-indexed
                
                # Set next question and correctness
                truth[i] = next_correct
                input_y[i, next_id - 1] = 1  # -1 because Python is 0-indexed
        
        # Apply compressed sensing if enabled
        if self.compressed_sensing:
            input_x = torch.mm(input_x, self.basis)
        
        return input_x, input_y, truth
    
    def accuracy(self, batch):
        n_steps = get_n_steps(batch)
        n_students = len(batch)
        
        # Set to evaluation mode
        training_mode = self.training
        self.eval()
        
        sum_correct = 0
        num_tested = 0
        
        with torch.no_grad():
            # Get initial state
            state = self.start(torch.zeros(n_students, 1))
            
            for k in range(1, n_steps + 1):
                input_x, input_y, truth = self.get_inputs(batch, k)
                mask = get_mask(batch, k)
                
                # Forward pass
                pred, _, state = self.layer(state, input_x, input_y, truth)
                
                # Calculate accuracy
                predictions = (pred > 0.5).float()
                correct = (predictions == truth).float()
                num_correct = (correct * mask).sum().item()
                
                sum_correct += num_correct
                num_tested += mask.sum().item()
        
        # Restore training mode
        self.train(training_mode)
        
        return sum_correct, num_tested    

#--------------------------
# Utility Functions
#--------------------------
def get_keyset(data):
    """Get indices of a list or keys of a dictionary"""
    if isinstance(data, list):
        return list(range(len(data)))
    else:
        return list(data.keys())

def shuffle(data):
    """Shuffle a list and return a new list"""
    result = data.copy()
    random.shuffle(result)
    return result

def get_mask(batch, k):
    """Create mask for valid positions in the batch"""
    mask = torch.zeros(len(batch))
    for i, ans in enumerate(batch):
        if k + 1 <= ans['n_answers']:
            mask[i] = 1
    return mask

def get_n_steps(batch):
    """Calculate the number of time steps in the batch"""
    return max(ans['n_answers'] for ans in batch) - 1

def get_n_tests(batch):
    """Calculate total number of test samples in a batch"""
    n_steps = get_n_steps(batch)
    n_students = len(batch)
    m = torch.zeros(n_students, n_steps)
    
    for i in range(1, n_steps + 1):
        mask = get_mask(batch, i)
        for j in range(n_students):
            m[j, i-1] = mask[j]
    
    return m.sum().item()

def get_total_tests(batches, max_steps=None):
    """Calculate total number of test samples across all batches"""
    total = 0
    for batch in batches:
        n_tests = get_n_tests(batch)
        if max_steps is not None and max_steps > 0:
            n_tests = min(max_steps, n_tests)
        total += n_tests
    return total

def semi_sorted_mini_batches(dataset, mini_batch_size, trim_to_batch_size=True):
    """Create mini-batches sorted by sequence length"""
    # Round down so that minibatches are the same size
    trimmed_ans = []
    if trim_to_batch_size:
        n_temp = len(dataset)
        max_num = n_temp - (n_temp % mini_batch_size)
        shuffled = shuffle(get_keyset(dataset))
        for i, s in enumerate(shuffled):
            if i < max_num:
                trimmed_ans.append(dataset[s])
    else:
        trimmed_ans = dataset.copy()
    
    # Sort answers by sequence length
    trimmed_ans.sort(key=lambda a: a['n_answers'])
    
    # Make minibatches
    mini_batches = []
    for j in range(0, len(trimmed_ans), mini_batch_size):
        mini_batch = []
        end_idx = min(j + mini_batch_size, len(trimmed_ans))
        for k in range(j, end_idx):
            mini_batch.append(trimmed_ans[k])
        mini_batches.append(mini_batch)
    
    # Shuffle minibatches
    shuffled_batches = []
    shuffled_indices = shuffle(get_keyset(mini_batches))
    for idx in shuffled_indices:
        shuffled_batches.append(mini_batches[idx])
    
    return shuffled_batches


def get_accuracy(rnn, data, mini_batch_size=None):
    """Calculate accuracy on test data"""
    n_test = math.floor(data.nTest / 50)
    mini_batches = semi_sorted_mini_batches(data.get_test_data(), n_test, False)
    sum_correct = 0
    sum_tested = 0
    
    for i, batch in enumerate(mini_batches):
        correct, tested = rnn.accuracy(batch)
        sum_correct += correct
        sum_tested += tested
        print(f'testMini {(i+1)/len(mini_batches):.4f}, {sum_correct/max(1, sum_tested):.4f}')
    
    return sum_correct / max(1, sum_tested)

#--------------------------
# Training
#--------------------------
def train_mini_batch(rnn, data, init_rate, decay_rate, mini_batch_size, blob_size, model_id):
    """Train the model using mini-batch gradient descent"""
    print('train')
    rate = init_rate
    epochs = 1
    
    # Set device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    rnn.to(device)
    
    while True:  # Infinite training loop (Ctrl+C to stop)
        start_time = time.time()
        mini_batches = semi_sorted_mini_batches(data.get_train_data(), blob_size, True)
        total_tests = get_total_tests(mini_batches)
        
        print(f"Epoch {epochs}: {len(mini_batches)} batches, {total_tests} total tests")
        
        sum_err = 0
        num_tests = 0
        done = 0
        rnn.optimizer.zero_grad()

        mini_tests = 0
        mini_err = 0
        
        for i, batch in enumerate(mini_batches):
            alpha = blob_size / total_tests
            
            # Forward and backward pass
            err, tests, max_norm = rnn.calc_grad(batch, rate, alpha)
            sum_err += err
            num_tests += tests
            
            done += blob_size
            mini_err += err
            mini_tests += tests
            
            # Update weights periodically
            if done % mini_batch_size == 0:
                rnn.optimizer.step()
                rnn.optimizer.zero_grad()
                print(f"Batch {i+1}/{len(mini_batches)}: {(i+1)/len(mini_batches):.4f}, Error: {sum_err/max(1, num_tests):.4f}")
                mini_err = 0
                mini_tests = 0
        
        # Calculate overall error and test accuracy
        avg_err = sum_err / max(1, num_tests)
        print("Evaluating on test data...")
        test_pred = get_accuracy(rnn, data)
        
        # Log results
        #file.write(f"{epochs}\t{avg_err}\t{test_pred}\t{rate}\t{time.perf_counter()}\n")
        #file.flush()
        
        print(f"Epoch {epochs}: Error={avg_err:.6f}, Accuracy={test_pred:.6f}, Rate={rate}, Time={time.time() - start_time:.2f}s")
        
        # Decay learning rate and save model
        rate = rate * decay_rate
        rnn.save(os.path.join(OUTPUT_ROOT, 'models', f"{model_id}_{epochs}"))
        epochs += 1
            

def run():
    """Main function to set up and run the training"""
    # Set random seed
    random.seed(int(time.time()))
    torch.manual_seed(int(time.time()))
    
    # Initialize parameters
    n_hidden = 400
    decay_rate = 1.0
    init_rate = 0.5
    mini_batch_size = 50
    blob_size = mini_batch_size
    
    # Create data loader to process all files
    data = Data(n_questions=N_QUESTIONS)
    
    rnn = RNN(params={
        'dropout': True,
        'n_hidden': n_hidden,
        'n_questions': data.n_questions,
        'max_grad': 100,
        'max_steps': data.n_questions,
    })

    name = "result_combined"
        
    train_mini_batch(rnn, data, init_rate, decay_rate, mini_batch_size, blob_size, name)

if __name__ == "__main__":
    run()