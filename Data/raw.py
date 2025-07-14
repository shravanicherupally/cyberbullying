import pandas as pd
import random
from datetime import datetime, timedelta

# 1500 bullying + 1500 non-bullying
bullying_comments = [
    "You are so stupid!", "Nobody likes you!", "Just shut up!", 
    "You're worthless!", "Such a loser!", "I hate you!", 
    "You're ugly and dumb!", "Go away loser!"
] * 200  # 8*200 = 1600

non_bullying_comments = [
    "Have a nice day!", "Good luck!", "You can do it!", 
    "That's awesome!", "So proud of you!", "You're amazing!", 
    "Keep going!", "Well done!"
] * 200  # 8*200 = 1600

comments = bullying_comments[:1500] + non_bullying_comments[:1500]
labels = [1]*1500 + [0]*1500

platforms = ["Twitter", "Facebook", "Instagram", "YouTube", "Reddit"]

random.seed(42)
data = []
for i in range(3000):
    comment = comments[i]
    label = labels[i]
    user_id = random.randint(1, 500)
    date = datetime.now() - timedelta(days=random.randint(0, 365))
    platform = random.choice(platforms)
    data.append([i+1, user_id, comment, date, platform, label])

df = pd.DataFrame(data, columns=["id", "user_id", "comment", "date", "platform", "label"])
df.to_csv("../Data/cyberbullying_detection_dataset.csv", index=False)

print(df['label'].value_counts())
print("✅ Balanced dataset created: saved to /Data/")
