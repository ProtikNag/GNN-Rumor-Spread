import numpy as np
from MisInfoSpread import MisInfoSpread
from experienceBuffer import ReplayBuffer
import torch
import random
import torch.nn as nn
import torch.optim
import copy

import torch.utils.data

memory = ReplayBuffer(10000)

eps = 1

misinfo = MisInfoSpread(10)

policy = misinfo.get_nnet_model()
target = misinfo.get_nnet_model()

optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
criterion = nn.MSELoss()

emax = 100

print("Initialization complete!")

target.load_state_dict(copy.deepcopy(policy.state_dict()))

batch_size = 32


def flatten(state):
    output = list()
    for val, adj in zip(state.node_states, state.adjacency_matrix):
        for i in adj:
            output.append(round(val * i, 2))
    return output


print("Training begins!")

t = 1
T = 5

for e in range(emax):
    print(f"\nStarting episode {e + 1} of {emax}")

    states = misinfo.generate_states(batch_size, 10)

    candidate_nodes_np = misinfo.find_neighbor_batch(states)

    while sum([len(candidate_nodes) for candidate_nodes in candidate_nodes_np]) > 0:
        blockernode = 0

        policy.eval()

        r = random.uniform(0, 1)
        if r < eps:
            blockernode_np = list()
            for candidate_nodes in candidate_nodes_np:
                if len(candidate_nodes) == 0:
                    blockernode_np.append(0)
                else:
                    blockernode_np.append(random.sample(candidate_nodes, 1)[0])

        else:
            output_tensor_np = [torch.FloatTensor(flatten(state)).view(1, -1) for state in states]

            expected_infected_np = [[policy(output_tensor)[0][node].detach().numpy() for node in candidate_nodes] for
                                    candidate_nodes, output_tensor in zip(candidate_nodes_np, output_tensor_np)]

            max_index_np = list()
            for expected_infected in expected_infected_np:
                if len(expected_infected) == 0:
                    max_index_np.append(0)
                else:
                    max_index_np.append(np.argmax(expected_infected))

            blockernode_np = list()
            for candidate_nodes, max_index in zip(candidate_nodes_np, max_index_np):
                if len(candidate_nodes) == 0:
                    blockernode_np.append(0)
                else:
                    blockernode_np.append(candidate_nodes[max_index])

        next_states, reward_np, done_np = misinfo.step_batch(states, blockernode_np)

        candidate_nodes_np = misinfo.find_neighbor_batch(next_states)

        for state, blockernode, reward, next_state, done in zip(states, blockernode_np, reward_np, next_states,
                                                                done_np):
            memory.push(state, blockernode, reward, next_state, done)

        data = memory.sample(batch_size)
        tarVal = list()

        data_array = np.array(data)

        eps = max(0.1, 1 - 0.9 * ((e * t) / emax))

        for i, val in enumerate(data):
            if val[4]:
                tarVal.append(val[2])
            else:
                output_tensor = torch.FloatTensor(flatten(val[3])).view(1, -1)
                values = [target(output_tensor)[0][node] for node in misinfo.find_neighbor(val[3])]
                if len(values) > 0:
                    max_val = torch.max(torch.tensor(values))
                    tarVal.append(val[2] + max_val)
                else:
                    tarVal.append(val[2])

        policy.train()
        optimizer.zero_grad()

        policy_outputs = list()
        for i, val in enumerate(data):
            output_tensor = torch.FloatTensor(flatten(val[3])).view(1, -1)
            action_value = policy(output_tensor)[0][val[1]]
            policy_outputs.append(action_value)

        policy_outputs_tensor = torch.stack(policy_outputs).unsqueeze(-1)

        tarVal_tensor = torch.tensor(tarVal, dtype=torch.float32).unsqueeze(-1)

        loss = criterion(policy_outputs_tensor, tarVal_tensor)

        loss.backward()
        optimizer.step()

        if t == T:
            print("Synchronizing target network with policy network...")
            t = 1
            target.load_state_dict(policy.state_dict())

        print(f"Loss: {loss.item()}")
        t += 1

        flag = 1
        for idx in range(len(done_np)):
            if not done_np[idx]:
                states[idx] = next_states[idx]
                flag = 0
            elif done_np[idx]:
                states[idx] = states[idx]

        if flag:
            break

    print(f"Episode {e + 1} finished")

print("\nTraining completed!")

print("Saving models...")
current_model_path = "./saved_models/current_model.pt"
target_model_path = "./saved_models/target_model.pt"

torch.save(policy.state_dict(), current_model_path)
torch.save(target.state_dict(), target_model_path)
print("Models saved successfully!")
