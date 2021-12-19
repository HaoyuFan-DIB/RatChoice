import numpy as np
import matplotlib.pyplot as plt
import random


class participant():
    def __init__(self, name=None, alpha=0.5, beta=1.0, updateboth=True):
        if name is None:
            self.name = "Participant"
        else:
            self.name = name

        self.both = updateboth
        self.alpha = alpha #learning rate
        self.beta = beta # degree of stochasticity, WTH is this???

        self.V_a = 0.5
        self.V_b = 0.5
        self.V_a_hist = []
        self.V_b_hist = []

    def _upgradeValues(self, selection, r):
        self.V_a_hist.append(self.V_a)
        self.V_b_hist.append(self.V_b)

        if selection == 1:
            self.V_a = self.V_a + self.alpha * (r - self.V_a)
            if self.both:
                self.V_b = self.V_b + self.alpha * (1 - r - self.V_b)

        if selection == 0:
            self.V_b = self.V_b + self.alpha * (r - self.V_b)
            if self.both:
                self.V_a = self.V_a + self.alpha * (1 - r - self.V_a)

    def _getProbability_A(self):
        z_A = self.beta * (self.V_a - self.V_b)
        P_A = 1 / (1 + np.exp(-z_A))
        return P_A

    def _getProbability_B(self):
        z_B = self.beta * (self.V_b - self.V_a)
        P_B = 1 / (1 + np.exp(-z_B))
        return P_B

    def _makeSelection(self):
        P_A = self._getProbability_A()
        P_B = self._getProbability_B()
        assert np.abs(P_A + P_B - 1.0) <= 0.001, \
            "Something wrong with P_A and P_B? Sum is {s:.2f}".format(s=P_A+P_B)
        # output: A = 1, B = 0
        select = random.uniform(0, 1)
        if select <= P_A:
            return 1
        else:
            return 0


class experiment():
    def __init__(self, r_a=0.8, r_b=0.2):
        r_a = np.array([0,r_a,1])
        r_a.sort()
        self.r_a = r_a[1]

        r_b = np.array([0, r_b, 1])
        r_b.sort()
        self.r_b = r_b[1]

    def _provideReward(self, respond):
        random_reward = random.uniform(0, 1)
        if respond == 1:
            if random_reward <= self.r_a:
                return 1
            else:
                return 0
        else:
            if random_reward <= self.r_b:
                return 1
            else:
                return 0


participantA = participant(name="Participant_A", alpha=0.1, beta=1.0, updateboth=False)  # the cleaver rat
participantB = participant(name="Participant_B", alpha=0.1, beta=3.0, updateboth=False)  # the dumb rat
exp = experiment(r_a=0.8, r_b=0.2)
n_trials = 400
A_Selections, A_rewards, B_Selections, B_rewards = [],[],[],[]
for i in range(n_trials):
    A_select = participantA._makeSelection()
    A_reward = exp._provideReward(A_select)
    A_Selections.append(A_select)
    A_rewards.append(A_reward)
    participantA._upgradeValues(A_select, A_reward)

    B_select = participantB._makeSelection()
    B_reward = exp._provideReward(B_select)
    B_Selections.append(B_select)
    B_rewards.append(B_reward)
    participantB._upgradeValues(B_select, B_reward)

markermap = ["x", "o"]
colormap = ["r", "b"]
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(np.arange(n_trials), np.array(participantA.V_a_hist), color="b", label="Value A")
ax1.plot(np.arange(n_trials), np.array(participantA.V_b_hist), color="r", label="Value B")
scatter_label_flag = [True, True]
scatter_labels = ["Selection = B", "Selection = A"]
for i, par in enumerate(zip(A_Selections, A_rewards)):
    s, r = par
    if scatter_label_flag[r]:
        ax1.scatter(i, (s-0.5)*1.1 + 0.5, marker=markermap[r], color=colormap[r], s=10, label=scatter_labels[r])
        scatter_label_flag[r] = False
    else:
        ax1.scatter(i,  (s-0.5)*1.1 + 0.5, marker=markermap[r], color=colormap[r], s=10)
ax1.set_xlabel(participantA.name + ", alpha={a:.2f}, beta={b:.2f}".format(a=participantA.alpha, b=participantA.beta))
ax1.set_ylim([-0.07,1.07])
ax1.set_xlim([0, n_trials])
ax1.grid()
ax1.legend()

ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(np.arange(n_trials), np.array(participantB.V_a_hist), color="b")
ax2.plot(np.arange(n_trials), np.array(participantB.V_b_hist), color="r")
for i, par in enumerate(zip(B_Selections, B_rewards)):
    s, r = par
    ax2.scatter(i, (s-0.5)*1.1 + 0.5, marker=markermap[r], color=colormap[r], s=10)
ax2.set_xlabel(participantB.name + ", alpha={a:.2f}, beta={b:.2f}".format(a=participantB.alpha, b=participantB.beta))
ax2.set_ylim([-0.07,1.07])
ax2.set_xlim([0, n_trials])
ax2.grid()
fig.tight_layout()
plt.show()



