""" DQN based PIRL implementation with PyTorch optimized for AWS Neuron
        This is a Neuron-optimized version of PIRL_torch.py
        Key optimizations:
        1. Traced forward pass for inference
        2. Optimized tensor operations for Neuron
        3. Batch processing optimizations
"""

__author__ = 'Neuron Optimization'
__email__ = 'neuron-optimized'

import os
import numpy as np
import random
import copy
from collections import deque
from tqdm import tqdm
import torch
import torch_neuronx
from torch.utils.tensorboard import SummaryWriter

# Import original functions that don't need modification
from PIRL_torch import agentOptions, pinnOptions, trainOptions, train

class PIRLagentNeuron:
    """Neuron-optimized version of PIRLagent"""
    
    def __init__(self, model, actNum, agentOp, pinnOp, trace_model=True): 

        # Agent Options
        self.actNum  = actNum
        self.agentOp = agentOp
        self.pinnOp  = pinnOp
        
        # Q-networks
        self.model = model
        self.optimizer = agentOp['OPTIMIZER']

        # Target Q-network 
        self.target_model = copy.deepcopy(self.model)

        # Neuron-specific optimizations
        self.traced_model = None
        self.traced_target_model = None
        self.trace_model = trace_model
        
        if trace_model:
            self._trace_models()

        # Replay Memory
        self.replay_memory = deque(maxlen=agentOp['REPLAY_MEMORY_SIZE'])

        # Initialization of variables
        self.epsilon = agentOp['EPSILON_INIT'] if agentOp['RESTART_EP'] == None else max( 
            self.agentOp['EPSILON_MIN'], 
            agentOp['EPSILON_INIT']*np.power(agentOp['EPSILON_DECAY'], agentOp['RESTART_EP'])
        )
        self.target_update_counter = 0
                
        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

    def _trace_models(self):
        """Trace models for Neuron optimization with multiple batch sizes"""
        print("Tracing models for Neuron optimization...")
        
        # Create traced models for common batch sizes
        self.traced_models = {}
        self.traced_target_models = {}
        
        # Extended batch sizes including large batches for optimal Neuron utilization
        batch_sizes = [1, 16, 32, 128, 512, 1024]  # Include large batches for throughput
        
        try:
            for batch_size in batch_sizes:
                print(f"  Tracing for batch size {batch_size} with high precision...")
                example_input = torch.randn(batch_size, self._get_input_size())
                
                # Trace the main model with autocast=none for higher precision
                self.model.eval()
                with torch.no_grad():
                    traced_model = torch_neuronx.trace(
                        self.model, 
                        example_input,
                        compiler_args=["--auto-cast", "none"]
                    )
                    self.traced_models[batch_size] = traced_model
                
                # Trace the target model with autocast=none
                self.target_model.eval()
                with torch.no_grad():
                    traced_target_model = torch_neuronx.trace(
                        self.target_model, 
                        example_input,
                        compiler_args=["--auto-cast", "none"]
                    )
                    self.traced_target_models[batch_size] = traced_target_model
            
            # Set default traced model (batch size 1)
            self.traced_model = self.traced_models.get(1)
            self.traced_target_model = self.traced_target_models.get(1)
                
            print(f"Models successfully traced for Neuron! Available batch sizes: {list(self.traced_models.keys())}")
            
        except Exception as e:
            print(f"Warning: Could not trace models for Neuron: {e}")
            print("Falling back to CPU execution")
            self.traced_models = {}
            self.traced_target_models = {}
            self.traced_model = None
            self.traced_target_model = None
            self.trace_model = False

    def _get_input_size(self):
        """Get input size from model"""
        for param in self.model.parameters():
            if len(param.shape) == 2:  # First linear layer
                return param.shape[1]
        return 16  # Default fallback

    def update_replay_memory(self, transition):
        """transition = (current_state, action, reward, new_state, done)"""
        self.replay_memory.append(transition)

    def get_qs(self, state):
        """Get Q-values with Neuron optimization"""
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
        if self.traced_model is not None:
            # Use traced model for faster inference
            with torch.no_grad():
                return self.traced_model(state_tensor).squeeze(0)
        else:
            # Fallback to regular model
            with torch.no_grad():
                return self.model(state_tensor).squeeze(0)
    
    def get_qs_batch(self, states):
        """Batch Q-value computation for better Neuron utilization"""
        states_tensor = torch.tensor(states, dtype=torch.float32)
        batch_size = states_tensor.shape[0]
        
        # Try to use traced model with matching batch size
        if self.traced_models and batch_size in self.traced_models:
            with torch.no_grad():
                return self.traced_models[batch_size](states_tensor)
        elif self.traced_model is not None and batch_size == 1:
            with torch.no_grad():
                return self.traced_model(states_tensor)
        else:
            # Fallback to regular model
            with torch.no_grad():
                return self.model(states_tensor)
    
    def get_epsilon_greedy_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.random() > self.epsilon:
            # Greedy action from Q network
            q_values = self.get_qs(state)
            action_idx = int(torch.argmax(q_values))
        else:
            # Random action
            action_idx = np.random.randint(0, self.actNum)  
        return action_idx

    def train_step(self, experience, is_episode_done):
        """Neuron-optimized training step"""

        ########################
        # Update replay memory
        self.update_replay_memory(experience)

        if len(self.replay_memory) < self.agentOp['REPLAY_MEMORY_MIN']:
            return

        ########################
        # Sample minibatch from experience memory
        minibatch = random.sample(self.replay_memory, self.agentOp['MINIBATCH_SIZE'])

        #######################
        # Calculate target y with batch processing
        current_states = np.array([transition[0] for transition in minibatch], dtype=np.float32)        
        new_current_states = np.array([transition[3] for transition in minibatch], dtype=np.float32)
        
        # Batch processing for better Neuron utilization
        current_qs_list = self.get_qs_batch(current_states).detach().numpy()
        
        # Use appropriate traced target model based on batch size
        batch_size = len(new_current_states)
        if self.traced_target_models and batch_size in self.traced_target_models:
            future_qs_list = self.traced_target_models[batch_size](
                torch.from_numpy(new_current_states)
            ).detach().numpy()
        elif self.traced_target_model is not None and batch_size == 1:
            future_qs_list = self.traced_target_model(
                torch.from_numpy(new_current_states)
            ).detach().numpy()
        else:
            future_qs_list = self.target_model(
                torch.from_numpy(new_current_states)
            ).detach().numpy()
        
        X = []  # feature set
        y = []  # label set (target y)

        for index, (current_state, action, reward, new_state, is_terminal) in enumerate(minibatch):
            if not is_terminal:
                max_future_q = future_qs_list[index].max()
                new_q = reward + self.agentOp['DISCOUNT'] * max_future_q
            else:
                new_q = reward

            current_qs = np.array(current_qs_list[index]) 
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        ##########################
        # Samples for PDE (optimized for batch processing)
        X_PDE, X_BDini, X_BDlat = self.pinnOp['SAMPLING_FUN'](self.replay_memory)

        # Batch convection and diffusion coefficient computation
        with torch.no_grad():
            pde_batch_size = X_PDE.shape[0]
            if self.traced_models and pde_batch_size in self.traced_models:
                Qsa = self.traced_models[pde_batch_size](torch.tensor(X_PDE, dtype=torch.float))
            elif self.traced_model is not None and pde_batch_size == 1:
                Qsa = self.traced_model(torch.tensor(X_PDE, dtype=torch.float))
            else:
                Qsa = self.model(torch.tensor(X_PDE, dtype=torch.float))
            Uidx_PDE = Qsa.argmax(1).numpy().reshape(-1, 1)               
        
        # Vectorized computation for better performance
        f = np.apply_along_axis(self.pinnOp['CONVECTION_MODEL'], 1, 
                               np.concatenate([X_PDE, Uidx_PDE], axis=1))
        A = np.apply_along_axis(self.pinnOp['DIFFUSION_MODEL'], 1, 
                               np.concatenate([X_PDE, Uidx_PDE], axis=1))

        ####################
        # DQN Loss (lossD)
        ####################
        y_pred = self.model(torch.from_numpy(X))
        y_trgt = torch.from_numpy(y)
        lossD = torch.nn.functional.mse_loss(y_pred, y_trgt)
    
        ####################
        # PDE loss (lossP) - Optimized gradient computation
        ####################
        if self.pinnOp['HESSIAN_CALC']: 
            X_PDE_tensor = torch.tensor(X_PDE, dtype=torch.float, requires_grad=True)
            Qsa = self.model(X_PDE_tensor)
            V = Qsa.max(1) 
            dV_dx = torch.autograd.grad(V.values.sum(), X_PDE_tensor, create_graph=True)[0]
        else: 
            X_PDE_tensor = torch.tensor(X_PDE, dtype=torch.float, requires_grad=True)
            Qsa = self.model(X_PDE_tensor)
            V = Qsa.max(1) 
            dV_dx = torch.autograd.grad(V.values.sum(), X_PDE_tensor, create_graph=True)[0]

        # Convection term
        conv_term = (dV_dx * torch.tensor(f, dtype=torch.float32)).sum(1)

        if self.pinnOp['HESSIAN_CALC']:
            lossP = torch.nn.functional.mse_loss(conv_term, torch.zeros_like(conv_term))             
        else:
            lossP = torch.nn.functional.mse_loss(conv_term, torch.zeros_like(conv_term))             
        
        ########################
        # Boundary loss (lossB) - Batch processing
        ########################
        # Terminal boundary
        y_bd_ini = self.model(torch.tensor(X_BDini, dtype=torch.float32)).max(1).values
        lossBini = torch.nn.functional.mse_loss(y_bd_ini, torch.ones_like(y_bd_ini))
        
        # Lateral boundary
        y_bd_lat = self.model(torch.tensor(X_BDlat, dtype=torch.float32)).max(1).values
        lossBlat = torch.nn.functional.mse_loss(y_bd_lat, torch.zeros_like(y_bd_lat))
        
        lossB = lossBini + lossBlat

        #####################
        # Total Loss function
        #####################
        Lambda = self.pinnOp['WEIGHT_PDE']
        Mu = self.pinnOp['WEIGHT_BOUNDARY']
        loss = lossD + Lambda*lossP + Mu*lossB      

        ############################
        # Update trainable variables
        ############################
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        if is_episode_done:
            #############################
            # Update target Q-function and decay epsilon            
            self.target_update_counter += 1

            if self.target_update_counter > self.agentOp['UPDATE_TARGET_EVERY']:
                self.target_model.load_state_dict(self.model.state_dict())
                
                # Re-trace target model if using Neuron
                if self.trace_model:
                    try:
                        self.traced_target_models = {}
                        batch_sizes = [1, 16, 32, 128, 512, 1024]
                        for batch_size in batch_sizes:
                            example_input = torch.randn(batch_size, self._get_input_size())
                            self.target_model.eval()
                            with torch.no_grad():
                                traced_target = torch_neuronx.trace(
                                    self.target_model, 
                                    example_input,
                                    compiler_args=["--auto-cast", "none"]
                                )
                                self.traced_target_models[batch_size] = traced_target
                        self.traced_target_model = self.traced_target_models.get(1)
                    except Exception as e:
                        print(f"Warning: Could not re-trace target model: {e}")
                
                self.target_update_counter = 0

            ##############################
            # Decay epsilon
            if self.epsilon > self.agentOp['EPSILON_MIN']:
                self.epsilon *= self.agentOp['EPSILON_DECAY']
                self.epsilon = max(self.agentOp['EPSILON_MIN'], self.epsilon)

    def load_weights(self, ckpt_dir, ckpt_idx=None):
        """Load weights and re-trace models"""
        if not os.path.isdir(ckpt_dir):         
            raise FileNotFoundError("Directory '{}' does not exist.".format(ckpt_dir))

        if not ckpt_idx or ckpt_idx == 'latest': 
            check_points = [item for item in os.listdir(ckpt_dir) if 'agent' in item]
            check_nums = np.array([int(file_name.split('-')[1]) for file_name in check_points])
            latest_ckpt = f'/agent-{check_nums.max()}'  
            ckpt_path = ckpt_dir + latest_ckpt
        else:
            ckpt_path = ckpt_dir + f'/agent-{ckpt_idx}'
            if not os.path.isfile(ckpt_path):   
                raise FileNotFoundError("Check point 'agent-{}' does not exist.".format(ckpt_idx))

        checkpoint = torch.load(ckpt_path)
        self.model.load_state_dict(checkpoint['weights'])
        self.target_model.load_state_dict(checkpoint['target-weights'])        
        self.replay_memory = checkpoint['replay_memory']
        
        # Re-trace models after loading weights
        if self.trace_model:
            self._trace_models()
        
        print(f'Agent loaded weights stored in {ckpt_path}')
        
        return ckpt_path

    def save_traced_models(self, save_dir):
        """Save traced models for deployment"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        if self.traced_model is not None:
            torch.jit.save(self.traced_model, os.path.join(save_dir, 'traced_model.pt'))
            print(f"Traced model saved to {save_dir}/traced_model.pt")
            
        if self.traced_target_model is not None:
            torch.jit.save(self.traced_target_model, os.path.join(save_dir, 'traced_target_model.pt'))
            print(f"Traced target model saved to {save_dir}/traced_target_model.pt")

    def benchmark_inference(self, num_samples=10000, batch_sizes=[1, 32, 128, 512, 1024, 2048]):
        """Benchmark inference performance with different batch sizes"""
        print("\n=== Neuron Inference Benchmarking ===")
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\nTesting batch size: {batch_size}")
            
            # Generate test data
            test_states = np.random.randn(num_samples, self._get_input_size())
            
            # Benchmark traced model
            if self.traced_model is not None:
                times = []
                for i in range(0, num_samples, batch_size):
                    batch = test_states[i:i+batch_size]
                    if len(batch) < batch_size:
                        continue
                        
                    start_time = time.time()
                    with torch.no_grad():
                        _ = self.traced_model(torch.tensor(batch, dtype=torch.float32))
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                avg_time = np.mean(times)
                throughput = batch_size / avg_time
                results[batch_size] = {
                    'avg_time': avg_time,
                    'throughput': throughput
                }
                
                print(f"  Traced model - Avg time: {avg_time*1000:.3f} ms, Throughput: {throughput:.1f} samples/sec")
            
            # Benchmark regular model for comparison
            times = []
            for i in range(0, num_samples, batch_size):
                batch = test_states[i:i+batch_size]
                if len(batch) < batch_size:
                    continue
                    
                start_time = time.time()
                with torch.no_grad():
                    _ = self.model(torch.tensor(batch, dtype=torch.float32))
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time_regular = np.mean(times)
            throughput_regular = batch_size / avg_time_regular
            
            print(f"  Regular model - Avg time: {avg_time_regular*1000:.3f} ms, Throughput: {throughput_regular:.1f} samples/sec")
            
            if self.traced_model is not None:
                speedup = avg_time_regular / results[batch_size]['avg_time']
                print(f"  Speedup: {speedup:.2f}x")
        
        return results


# Convenience function to create Neuron-optimized agent
def create_neuron_agent(model, actNum, agentOp, pinnOp, trace_model=True):
    """Create a Neuron-optimized PIRL agent"""
    return PIRLagentNeuron(model, actNum, agentOp, pinnOp, trace_model)