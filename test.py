# from adaptive_nav.openenv_client import NavEnvClient
# from adaptive_nav.models import NavAction

# client = NavEnvClient("ws://localhost:8000")

# obs = client.reset()

# print(obs)

# from adaptive_nav.models import NavAction

# result = client.step(NavAction(action_id=3))
# print(result)
# print(result.observation, result.reward)





# from adaptive_nav.openenv_client import NavEnvClient
# from adaptive_nav.models import NavAction
# import random

# ACTION_NAMES = ["up", "down", "left", "right", "interact", "wait"]

# client = NavEnvClient("ws://localhost:8000")

# result = client.reset()
# print("reset ok")

# for i in range(10):
#     action_id = random.randint(0, 5)
#     result = client.step(NavAction(action_id=action_id))
#     print(i, ACTION_NAMES[action_id], result.reward, result.done)
#     if result.done:
#         break



from adaptive_nav.openenv_client import NavEnvClient
from adaptive_nav.models import NavAction
# from adaptive_nav.openenv_client import NavEnvClient
import random

ACTION_NAMES = [
    "up",
    "down",
    "left",
    "right",
    "interact",
    "wait",
]

client = NavEnvClient("https://harishchaurasia-adaptive-nav-openenv.hf.space")

result = client.reset()
print("reset ok")

for i in range(10):
    action_id = random.randint(0, 5)
    result = client.step(NavAction(action_id=action_id))
    print(i, ACTION_NAMES[action_id], result.reward, result.done)
    if result.done:
        break