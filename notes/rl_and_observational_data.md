# Why reinforcement learning does not fit this panel (Session 8, 14 Nov 2026)

**Course link:** Machine Learning Specialization, Course 3 Week 3 (reinforcement learning: states, actions, rewards, return and discount factor, policy, Markov decision process, Q(s, a), the Bellman equation, deep Q-networks with a replay buffer, ε-greedy exploration, mini-batches, soft updates).

## What reinforcement learning needs

Reinforcement learning learns by **trying actions and seeing what happens**. An agent is in a state, takes an action, receives a reward and lands in a new state, then does this again many times, sometimes choosing a random action on purpose (ε-greedy) to find out what other actions would have given. The lunar lander in the course crashes thousands of times in a simulator before it learns to land.

So it needs three things:
1. **An environment you can act in:** the agent chooses the action, and the environment answers.
2. **Many tries, including the bad ones:** exploring actions that look worse is how it learns that they are worse.
3. **A reward it can observe for each action it takes.**

## Why my panel does not give that

- **It is history, not an environment.** Each country-year happened once, with the budget choice the government actually made. I cannot go back to Greece in 2010 and try *not* consolidating to see what TFP would have done. Every "action" in the data was chosen by a government, not by an agent that could explore.
- **The actions were not random.** Governments consolidated when they were in trouble (high debt, crises, market pressure). An algorithm that learns "consolidating is followed by low growth" from this history would partly be learning "being in trouble is followed by low growth". That is the selection bias from the Mixtape: the states that led to the action also affect the outcome.
- **The reward is the very thing we want to explain.** In RL the reward is given: the simulator says "+100 for landing". Here the reward would be future TFP growth, and the whole question is how much of it the consolidation *caused*. Using it as a reward assumes we already know the answer.
- **Too little data for trial and error.** 26 countries × 30 years, about 100 consolidation episodes. A deep Q-network needs thousands or millions of transitions it can generate itself.
- **Countries affect each other.** A consolidation in one EU country spills over to its trading partners, so one country-year's "reward" is not only the result of its own action (SUTVA does not hold).

An RL agent trained on this panel would not learn the effect of consolidating. At best it would learn to imitate past governments, with all their reasons mixed in.

## What would be needed instead

The question "what would growth have been without the consolidation?" is a **causal** question, about outcomes we never see (potential outcomes). That needs causal inference, not reinforcement learning:
- **A design that makes consolidating and non-consolidating years comparable:** fixed effects and controls as in my thesis, local projections, and checks for pre-trends (the thesis finds that consolidating countries were already diverging before the shock).
- **Sources of variation that are closer to random:** for example consolidations chosen for reasons unrelated to the current state of the economy (narrative identification), or instruments.
- **Estimands that match the data:** the thesis estimates are closest to an ATT, the effect for the consolidations that actually happened, not the effect of a policy rule applied everywhere.

Reinforcement learning would become relevant only with a trusted **simulator** of the economy (a structural model), where an agent could try policy rules safely. The answer would then be only as good as the simulator, and building that simulator is itself the causal-modelling problem.

**In one line:** reinforcement learning answers "what should I do, given an environment I can try things in"; my panel can only answer "what happened, and why", and the "why" needs causal inference.
