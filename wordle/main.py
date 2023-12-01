from tkinter.tix import ExFileSelectBox
import gym
import gym_wordle
import time
import numpy as np
import random

from gym_wordle.utils import to_english, to_array, get_words
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
import os

# import rl
# from rl.agents import DQNAgent
# from rl.policy import BoltzmannQPolicy
# from rl.memory import SequentialMemory

env = gym.make("Wordle-v0")


get_words('solution',True)
get_words('guess',True)

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

#policy definition
def build_model(states, actions):
    #model = Sequential()    

    inputs = layers.Input(shape=( states))

    number_input_nodes = 512
    layer1 = layers.Dense(number_input_nodes, activation="relu")(inputs) 
    layer2 = layers.Dense(1023, activation="linear")(layer1)

    #layer3 = layers.Dense(min(number_input_nodes+actions, 2*number_input_nodes-1), activation="linear")(layer1)
    layer3 = layers.Dense(1023, activation="relu")(layer2)
    layer4=  layers.Flatten()(layer3)
    layer5 = layers.Dense(130, activation='linear')(layer4)

    action = layers.Reshape(actions)(layer5)

    return keras.Model(inputs=inputs, outputs=action)

    # model.add(Dense(512, activation='relu', input_shape=(1, states[0],1)))
    
    # #model.add(Dense(24, activation='relu'))
    # model.add(Dense(actions, activation='linear'))
    # model.add(Flatten())

    # return model

# Train the model after 4 actions
update_after_actions = 3

#Hyperparameters
alpha = 0.1
seed = 42
gamma = 0.7  # Discount factor for past rewards 0.99
epsilon = 0.8    # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
    epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
batch_size = 64  # Size of batch taken from replay buffer

# Number of frames to take random action and observe output
epsilon_random_loop = 100
# Number of frames for exploration
epsilon_greedy_loop = 20000000

loop_count = 0

all_epochs = []
all_penalties = []

learning_rate_ = 0.02
slowdown_rate = update_after_actions* 50

# In the Deepmind paper they use RMSProp however then Adam optimizer
# improves training time
optimizer = keras.optimizers.SGD(learning_rate=learning_rate_,  decay=0.001, momentum=0.2)
 #Adam(learning_rate=learning_rate_, clipnorm=1.0)#. #

# Experience replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
# episode_reward_history = []
# running_reward = 0
episode_count = 0
loop_count = 0
min_reward = 5


last_100_predict = []
number_predict = 0
pre_target = 0


success_predicted = []

max_memory_length = 100000



# How often to update the target network
update_target_network = 1000
# Using huber loss for stability
loss_function = keras.losses.Huber()

states = env.observation_space.shape
# print(states)
actions = (5,26)

# The first model makes the predictions for Q-values which are used to
# make a action.
model = build_model(states,actions)
#model.fit(verbose=0)
# Build a target model for the prediction of future rewards.
# The weights of a target model get updated every 10000 steps thus when the
# loss between the Q-values is calculated the target Q-value is stable.
model_target = build_model(states,actions)

#model.compile(optimizer='sgd', loss=tf.keras.losses.Poisson())
try:
    model.load_weights('./checkpoints/my_checkpoint5')
    model_target.load_weights('./checkpoints/my_checkpoint5')
except:
    model.save_weights('./checkpoints/my_checkpoint5')
    print("no saved model")

ave_loss =  0
last_lost_rate = 0



for episode in range(1000000):
    # reset env
    state = env.reset()

 #   if env.solution in success_predicted: # already solve try another
 #       print("Get a A new one!!")
 #       state = env.reset()

    done = False
    is_predict  = False
    is_first_word = True
    epochs, penalties, episode_reward,max_reward = 0, 0, 0 , 0

    last_reward = 0
    while not done:
        loop_count +=1 
 
        is_predict = False
        if is_first_word:
            action = to_array("crane")#.flatten()
        else:
            if 0.2 > np.random.rand(1)[0] and env.round >= 3:
                picked_word = to_english(env.solution_space[env.solution])
                
                ramdon_action = to_array(picked_word)#.flatten() # random selection from dictionary
                #print("tell answeer:",to_english(ramdon_action), "            action:", ramdon_action)
            else:
                ramdon_action = env.action_space.sample().flatten() # pur ramdom
                #print("ramdon_action answeer:",to_english(ramdon_action), "            action:", ramdon_action)

            
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)

            #print("action_probs:",action_probs)

            # Take best action
            #tf.config.run_functions_eagerly(True)
            if tf.executing_eagerly():

                #print("action_probs",action_probs[0][0])
                predicted_action = [tf.argmax(action_probs[0][0]).numpy(),
                            tf.argmax(action_probs[0][1]).numpy(),
                            tf.argmax(action_probs[0][2]).numpy(), 
                            tf.argmax(action_probs[0][3]).numpy(),
                            tf.argmax(action_probs[0][4]).numpy()]

            else:
                print('TF not eagerly!!')
                predicted_action = [tf.argmax(action_probs[0][0]).eval(),
                        tf.argmax(action_probs[0][1]).eval(),
                        tf.argmax(action_probs[0][2]).eval(), 
                        tf.argmax(action_probs[0][3]).eval(),
                        tf.argmax(action_probs[0][4]).eval()]
                    
            action=[]
            if ( episode < epsilon_random_loop or epsilon > np.random.rand(1)[0]) :
                for i in range(5):
                    if ( episode < epsilon_random_loop or epsilon*0.8 > np.random.rand(1)[0]) :# or env.round ==0: #*(3/2-1/(env.round+1))
                        action.append(ramdon_action[i])
                    else:
                        action.append(predicted_action[i])
            else:
                action= predicted_action
                is_predict = True

        #     #print ("target:",to_english(env.solution_space[env.solution]))
        #     if 0.2 > np.random.rand(1)[0] and env.round >= 3:
        #         picked_word = to_english(env.solution_space[env.solution])
                
        #         action = to_array(picked_word)#.flatten() # random selection from dictionary
        #         #print("tell answeer:",to_english(action), "            action:", action)
        #     else:
        #         action = env.action_space.sample().flatten() # pur ramdom

        #     #print("Random:",to_english(action), "            action:", action)

        # else:

        #     is_predict = True
        #     # Predict action Q-values
        #     # From environment state

        #     #print("action",action)

        # Decay probability of taking random action
        epsilon -= epsilon_interval / epsilon_greedy_loop
        epsilon = max(epsilon, epsilon_min)

        state_next, reward, done, info = env.step(action)
        state_next = np.array(state_next)
        

        # episode_reward += reward
        # max_reward = max(max_reward,reward)
        if is_predict:
            #print("target:",to_english(env.solution_space[env.solution]),"Predict:",to_english(action), "Predict:",action,"            reward:", reward, " round:", env.round,"epsilon:",epsilon)
            if env.solution != pre_target:
                number_predict = number_predict+1
                pre_target = env.solution
                last_100_predict.append (False)
            if np.sum(reward) >= 50 and number_predict >0:
                #print(env.solution,"!=", pre_target)
                last_100_predict[number_predict-1] = True
                if not env.solution in success_predicted:
                    success_predicted.append(env.solution)
                    print ("Newly Solved !!!!!!!!!!!!!!!!!!!!!!!! at round:",env.round," size:",len(success_predicted))
                #else:
                    #print ("Solved !!!!!!!!!!!!!!!!!!!!!!!! at round:",env.round," size:",len(success_predicted))
        # else:
        #      print("target:",to_english(env.solution_space[env.solution]),"Radom:",to_english(action), "            reward:", reward, " round:", env.round,"epsilon:",epsilon)

  
        if env.round != 1 :
        # Save actions and states in replay buffer
            action_history.append(action)
            state_history.append(state)
            state_next_history.append(state_next)
            done_history.append(done)
            rewards_history.append(reward)
            last_reward = reward
            # print("action:",action)
            # print("reward:", reward)
        state = state_next
        is_first_word = False

        # Update every sixth frame and once batch size is over 32
        if loop_count % update_after_actions == 0 and len(done_history) > batch_size*10:

            #print("memorize!!!",loop_count,update_after_actions, len(done_history))
            # Get indices of samples for replay buffers
            indices = np.random.choice(range(len(done_history)), size=batch_size)

            # Using list comprehension to sample from replay buffer
            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])
            rewards_sample = [rewards_history[i] for i in indices]
            action_sample = [action_history[i] for i in indices]
            done_sample = tf.convert_to_tensor(
                [float(done_history[i]) for i in indices]
            )

            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            future_rewards = model_target.predict(state_next_sample, verbose = 0)

            # Q value = reward + discount factor * expected future reward
            #print("future_rewards1",future_rewards)

            future_rewards = gamma * tf.reduce_max(
                future_rewards, axis=2
            )
            #print("future_rewards",future_rewards)
            #print("done_sample1",done_sample)

            #print("done_sample",np.rot90(np.tile(done_sample, [5,1]),3))

            # If final frame set the future reward to 0
            future_rewards = future_rewards * (1 - np.rot90(np.tile(done_sample, [5,1]),3)) 

            #print("rewards_sample",rewards_sample)

            updated_q_values = rewards_sample + future_rewards


            #NEW change
            #future_rewards[0][action_sample] = updated_q_values
            #model.fit(state_sample,future_rewards,epochs=1)
            # If final frame set the last value to -1
            #updated_q_values = updated_q_values * (1 - done_sample) - done_sample

            # # Create a mask so we only calculate loss on the updated Q-values
            masks = tf.one_hot(action_sample, 26)
            #print ("action_sample",action_sample)
            #print("masks",masks)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = model(state_sample)
                #print("q_values",q_values)

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_value_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=2)
                #print("q_value_action",q_value_action)
                #print("updated_q_values",updated_q_values)
                # Calculate loss between new Q-value and old Q-value
                loss = loss_function(updated_q_values, q_value_action) #q_value_action

                if ave_loss == 0:
                    ave_loss  = loss
                else:
                    ave_loss = (ave_loss *19+loss )/20


            # Backpropagation
            grads = tape.gradient(loss, model.trainable_variables)
            if loop_count % 80 == 0:
                print("loss=========", loss.numpy(),"ave_loss =========", ave_loss.numpy(),"learning_rate_", learning_rate_)
                #print(">>>rewards_sample:",rewards_sample, "updated_q_values:",updated_q_values, "q_value_action",q_value_action )
                model.save_weights('./checkpoints/my_checkpoint5')
            #if loop_count % slowdown_rate == 0:


                # if ave_loss >=1.6:
                #     learning_rate_ = 0.05
                #     #slowdown_rate = update_after_actions* 1500
                #     print ("Reset the learning rate!!!!!!!!!",learning_rate_)
                #     last_lost_rate == 0.0
                # else:
                #learning_rate_ = 0.05/ (1+(loop_count / slowdown_rate)/5)

                #optimizer = keras.optimizers.SGD(learning_rate=learning_rate_,  decay=1e-6, momentum=0.9, nesterov=True)
            
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if loop_count % update_target_network == 0:

            # update the the target network with new weights
            model_target.set_weights(model.get_weights())
            # Log details
            template = "running  episode {}, frame count {}"
            print(template.format( episode_count, loop_count))

        # Limit the state and reward history
        if len(rewards_history) > max_memory_length:

            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]

        if number_predict > 1000:
            number_predict =number_predict - 1
            del last_100_predict[:1]

    # Update running reward to check condition for solving
    # episode_reward_history.append(episode_reward)
    # if len(episode_reward_history) > 100:
    #     del episode_reward_history[:1]
    # running_reward = np.mean(episode_reward_history)

    episode_count += 1
    if episode%1000 == 0:
        print('Episode:{} Score:{} last:{}'.format(episode, max_reward, reward))
    if episode%100 == 0:
        detected_= last_100_predict.count(True)
        print('###################detected:{} Out of:{} '.format(detected_, number_predict),"rate:", detected_/(number_predict+1)*100)


for layer in model.layers:
    weights = layer.get_weights()
    print (">>>",weights)

env.close()

# q_table = np.zeros([(3^5)^26, env.action_space.n])

# def state_to_index(state):
#         index =0
#         print("current state", state)

#         for i in range(26):
#             for j in range(5):
#                 if state[i,j] != 0:
#                     index += state[i,j]*4**j*1024**i
#         return index

#def build_model(states, actions):

    #model = Sequential()    


    # model.add(Dense(24, activation='relu', input_shape=(1, states[0],1)))
    
    # model.add(Dense(24, activation='relu'))
    # model.add(Dense(actions, activation='linear'))
    # model.add(Flatten())

    # return model

# def build_agent(model, actions):
#     policy = BoltzmannQPolicy()
#     memory = SequentialMemory(limit=50000, window_length=1)
#     dqn = DQNAgent(model=model, memory=memory, policy=policy, 
#                   nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
#     return dqn

# states = env.observation_space.shape
# print(states)
# actions = env.action_space.n

# model = build_model(states, actions)
# model.summary()
# dqn = build_agent(model, actions)

# # dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
# # dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)
# # scores = dqn.test(env, nb_episodes=100, visualize=True)
# # print(np.mean(scores.history['episode_reward']))
# # model.save('model.h5')


# dqn.compile(Adam(lr=1e-3), metrics=['mae'])
# dqn.fit(env, nb_steps=60000, visualize=False, verbose=1)

# results = dqn.test(env, nb_episodes=150, visualize=False)
# print(np.mean(results.history['episode_reward']))
