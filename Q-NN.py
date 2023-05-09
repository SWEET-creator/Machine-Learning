import csv
import numpy as np
import matplotlib.pyplot as plt

def csv_reader(file_name):
    cost = []
    name = []
    data_set = []
    f = open(file_name,'r')
    reader = csv.reader(f)
    i = 0
    for row in reader:
        data_set.append(row)
        i += 1
    j = 0
    for row in data_set:
        if (j < int(i/2)):
            row = [int(s) for s in row]
            cost.append(row)
        else :
            name.append(row)
        j += 1
    return cost, name

reward_data, state_data = csv_reader("sample.csv")
print(reward_data, state_data)

discount_fac = 0.9
learning_rate = 0.1
# random: epsilon = 1, greedy: epsilon = 0
epsilon = 0.2
L = 5000
l = 1 # length taking average
T = 100

action_dictionaly = {"Up":0,"Down":1,"Right":2,"Left":3}

q_value = {}

def _init_q_values():
    for i in state_data:
        for s in i:
            q_value[s] = [0.0 for a in action_dictionaly]

def print_state_reward(s = state_data,r = reward_data):
    x = y = 0
    for i in s:
        y = 0
        for j in i:
            print("(", x, ",", y, ")", "name:", j, "reward:", r[x][y])
            y += 1
        x += 1

def get_position_from_name(name):
    for i in range(len(state_data)):
        if name in state_data[i]:
            start_index = state_data[i].index(name)
            return [i,start_index]
    print("error "+name+" is nothing")
    return False

def get_name_from_position(position):
    return state_data[position[1]][position[0]]

def is_valid_action(s, action):
    x_prime = x = s[0]
    y_prime = y = s[1]
    if action == action_dictionaly["Up"]:
        y_prime = y - 1
    elif action == action_dictionaly["Down"]:
        y_prime = y + 1
    elif action == action_dictionaly["Right"]:
        x_prime = x + 1
    elif action == action_dictionaly["Left"]:
        x_prime = x - 1

    if len(state_data) <= y_prime or 0 > y_prime:
        return False
    elif len(state_data[0]) <= x_prime or 0 > x_prime:
        return False
    return True

# policy π
def policy_value(s, action):
    if action == action_dictionaly["Up"]:
        return 0.1
    elif action == action_dictionaly["Right"]:
        return 0.8
    elif action == action_dictionaly["Left"]:
        return 0.1
    else:
        return 0.0

#State_transition_probability
def P_value(s_prime, s, action):
    if (is_valid_action(s, action)):
        return 1.0
    else :
        return 0.0

def r_value(s):
    return reward_data[s[1]][s[0]]

def s_prime_value(s, action):
    if is_valid_action(s, action):
        x_prime = x = s[0]
        y_prime = y = s[1]
        if action == action_dictionaly["Up"]:
            y_prime = y - 1
        elif action == action_dictionaly["Down"]:
            y_prime = y + 1
        elif action == action_dictionaly["Right"]:
            x_prime = x + 1
        elif action == action_dictionaly["Left"]:
            x_prime = x - 1
        return [x_prime, y_prime]
    else:
        return s

#state-value function
def V_value(s,t):
    V_policy_st_is_s = 0.0
    for a in action_dictionaly.values():
        if is_valid_action(s, a):
            s_prime = s_prime_value(s, a)
            V_policy_st_is_s += \
                policy_value(s, a) * P_value(s_prime, s, a) * (r_value(s_prime) + discount_fac * V_value(s_prime,t))
            print(get_name_from_position(s_prime), policy_value(s, a), P_value(s_prime, s, a), r_value(s_prime)+ discount_fac * V_value(s_prime,t))
        else:
            pass
    return V_policy_st_is_s


def Q_larning(s, a):
    s_name = get_name_from_position(s)
    q = q_value[s_name][a]

    s_prime = s_prime_value(s, a)
    s_prime_name = get_name_from_position(s_prime)

    max_q = max(q_value[s_prime_name])

    q_value[s_name][a] = q + \
            (learning_rate * (r_value(s_prime) + (discount_fac * max_q) - q))

def act(s):
    # ε-greedy法
    if np.random.uniform() < epsilon:
        action = np.random.randint(0, len(action_dictionaly))
    else:
        action = np.argmax(q_value[get_name_from_position(s)])
    return action

def get_average_in_some_range(X, l):
    average_X = []
    x = []
    k = 0
    for i in X:
        if k == l:
            k = 0
            average_X.append(np.average(x))
            x = []
        x.append(i)
        k += 1
    average_X.append(np.average(x))
    return average_X

def graphical(average_reward):
    plt.plot(np.arange(1,L/l+1), average_reward)
    plt.xlabel("trial")
    plt.ylabel("reward")
    plt.show()

def main():
    _init_q_values()
    sum_reward = []
    for i in range(L):
        print(i+1,"回目")
        # initialize
        s = get_position_from_name("aa")
        t = 1
        reward = []
        route = []

        while(t <= T):
            for a in action_dictionaly.values():
                Q_larning(s,a)
            reward.append(r_value(s))
            route.append(get_name_from_position(s))
            action = act(s)
            s = s_prime_value(s, action)
            t += 1
        sum_reward.append(sum(reward))
        print(route)
    average_reward = get_average_in_some_range(sum_reward, l)
    graphical(average_reward)


main()